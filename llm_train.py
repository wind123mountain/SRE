from arguments import Arguments
from teacher_llm import Teacher, TeacherOutput
from student import StudentCausalModel, StudentOutput
from data_utils import LLMDataset, LLMDataCollator
from loss import cosine_token_weight_loss, derivative_loss
import math
from transformers import AutoTokenizer
from torch import nn
import torch.nn.functional as F
import torch

from torch import optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_scheduler
from evaluator import Evaluator



def load_tokenizer(model_type, path, kwargs):        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, **kwargs)
    if model_type in ["gpt2", "opt", "llama", "gptj", "llama2", "mistral", "tinyllama", "minicpm"]:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif model_type == "qwen":
        # tokenizer.pad_token_id = 151646
        tokenizer.eos_token_id = 151643
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print('tokenizer unknow')
    
    return tokenizer

from typing import Type
from torch.utils.data import DataLoader, Dataset
from torch import nn


def get_token_mapping(s_tokenizer, t_tokenizer, device):
    t_vocab = t_tokenizer.get_vocab()
    s_vocab = s_tokenizer.get_vocab()
    t_id_mapping = []
    s_id_mapping = []
    for s_token, s_token_id in s_vocab.items():
        if s_token in t_vocab:
            s_id_mapping.append(s_token_id)
            t_id_mapping.append(t_vocab[s_token])

    return torch.tensor(s_id_mapping, device=device), torch.tensor(t_id_mapping, device=device)

class Trainer:
    def __init__(self, student: StudentCausalModel, student_model_type: str, teacher_model_type: str,
                 args: Arguments, teacher_model: Teacher = None,
                 hidden_loss_weights = [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 8, 10]):
        super().__init__()

        self.student = student.train()
        self.teacher_model = teacher_model

        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.mse_loss = nn.MSELoss(reduction='mean')
        
        self.args = args
        self.args.p = max(args.p, 1e-5)

        self.alpha = args.hard_label_loss_weight
        self.temperature = args.temperature

        self.step = 0

        sum_hidden_loss_weights = sum(hidden_loss_weights)
        self.hidden_loss_weights = [w / sum_hidden_loss_weights for w in hidden_loss_weights]

        self.train_loader, self.val_loader, self.test_loader = self.get_data_loader(args, student_model_type, teacher_model_type)

        self.s_vocab_size = self.student.model.model.config.vocab_size
        self.student_loss_function = self.student.model.model.loss_function

        
        self.teacher_lm_head = nn.Linear(self.teacher_model.model.lm_head.in_features,
                                         self.teacher_model.model.lm_head.out_features,
                                         bias=(self.teacher_model.model.lm_head.bias is not None)
                                        ).to(self.student.device)
        self.teacher_lm_head.load_state_dict(self.teacher_model.model.lm_head.state_dict())
        for p in self.teacher_lm_head.parameters():
            p.requires_grad = False

        self.s_id_mapping, self.t_id_mapping = get_token_mapping(self.student_tokenizer, 
                                                                 self.teacher_tokenizer, 
                                                                 device=self.student.device)
        
        # ===== DS-KD with CMA modules =====
        self.s_hidden_size = self.student.model.model.config.hidden_size
        self.t_hidden_size = self.teacher_model.model.config.hidden_size

        self.dskd_index_projector_s2t = nn.Linear(
            2 * self.s_hidden_size,
            2 * self.t_hidden_size
        ).to(self.student.device)

        self.dskd_value_projector_t2s = nn.Linear(
            self.t_hidden_size,
            self.s_hidden_size
        ).to(self.student.device)

        self.dskd_value_projector_s2t = nn.Linear(
            self.s_hidden_size,
            self.t_hidden_size
        ).to(self.student.device)

        self.student_pad_id = self.student_tokenizer.pad_token_id
        self.teacher_pad_id = self.teacher_tokenizer.pad_token_id

        if self.student_pad_id is None:
            self.student_pad_id = self.student_tokenizer.eos_token_id
        if self.teacher_pad_id is None:
            self.teacher_pad_id = self.teacher_tokenizer.eos_token_id

    def get_data_loader(self, args: Arguments, student_model_type: str, teacher_model_type: str):
        self.student_tokenizer = load_tokenizer(student_model_type, args.student_tokenizer, 
                                                args.load_student_tokenizer_kwargs)
        self.teacher_tokenizer = load_tokenizer(teacher_model_type, args.teacher_tokenizer, 
                                                args.load_teacher_tokenizer_kwargs)

        train_dataset = LLMDataset(args.train_data, self.student_tokenizer, 
                                   self.teacher_tokenizer, args.max_len // 2)

        train_collate = LLMDataCollator(self.student_tokenizer, self.teacher_tokenizer,
                                       do_train=True, max_len = args.max_len,
                                       pad_to_multiple_of = args.pad_to_multiple_of,
                                       return_tensors = 'pt', padding = True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, collate_fn=train_collate)

        return train_loader, None, None


    def get_teacher_eval(self, inputs):
        outputs = self.teacher_model.decode(inputs)
  
        if outputs.hidden_states is not None:
            outputs.hidden_states = outputs.hidden_states.to(self.student.device, non_blocking=True)
            
        if outputs.span_weights is not None:
            outputs.span_weights=outputs.span_weights.to(self.student.device, non_blocking=True)

        return outputs

    def soft_label_distill_loss(self, student_logits, teacher_logits, distill_temperature = 2.0):
        
        student_probs = F.log_softmax(student_logits / distill_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / distill_temperature, dim=-1)

        loss = F.kl_div(student_probs, teacher_probs, reduction='batchmean')

        return loss

    def js_div(self, student_logits, teacher_logits,):
        p = F.softmax(student_logits, dim=-1)
        q = F.softmax(teacher_logits, dim=-1)

        m = 0.5 * (p + q)
        eps = 1e-6
        p = p.clamp(min=eps)
        q = q.clamp(min=eps)
        m = m.clamp(min=eps)

        js = 0.5 * (
            F.kl_div(m.log(), p, reduction='none').sum(dim=-1) +
            F.kl_div(m.log(), q, reduction='none').sum(dim=-1)
        )

        mask = (student_logits.abs().sum(dim=-1) != 0).float()

        js = (js * mask).sum() / mask.sum().clamp(min=1e-5)

        return js / math.log(2)

    
    def masked_kl_loss(self, student_logits, teacher_logits, mask, temperature=2.0):

        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction="none"
        ).sum(dim=-1)

        mask = mask.float()
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)

        return loss

    def forward_kl(self, logits, teacher_logits, mask):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = mask.float()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    def reverse_kl(self, logits, teacher_logits, mask):
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
        prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
        prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = mask.float()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    def skewed_forward_kl(self, logits, teacher_logits, lam=0.01):
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        mixed_probs = lam * teacher_probs + (1-lam) * student_probs
        mixed_logprobs = torch.log(mixed_probs)
        
        mask = (logits.abs().sum(dim=-1) != 0).float()
        inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

        prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
        return distil_loss

    def dskd_with_cma(
        self,
        student_outputs,
        teacher_outputs,
        student_inputs,
        teacher_inputs,
        labels,
        temperature=2.0,
    ):
        device = self.student.device

        s_input_ids = student_inputs["input_ids"].to(device)
        t_input_ids = teacher_inputs["input_ids"].to(device)

        labels = labels.to(device)

        teacher_labels = teacher_inputs.get("labels", None)
        if teacher_labels is None:
            teacher_labels = t_input_ids.clone()
            if "attention_mask" in teacher_inputs:
                teacher_labels[teacher_inputs["attention_mask"].to(device) == 0] = -100
        else:
            teacher_labels = teacher_labels.to(device)

        s_mask = labels.ne(-100)
        t_mask = teacher_labels.ne(-100)

        # formal ids để tránh index lỗi ở vị trí -100 / pad
        s_formal_target = torch.where(s_mask, labels, torch.zeros_like(labels))
        t_formal_target = torch.where(t_mask, teacher_labels, torch.zeros_like(teacher_labels))

        s_formal_input = torch.where(
            s_mask,
            s_input_ids[:, :labels.size(1)],
            torch.zeros_like(labels)
        )

        t_formal_input = torch.where(
            t_mask,
            t_input_ids[:, :teacher_labels.size(1)],
            torch.zeros_like(teacher_labels)
        )

        s_embed_layer = self.student.model.model.get_input_embeddings()
        t_embed_layer = self.teacher_model.model.get_input_embeddings()

        with torch.no_grad():
            s_input_embeds = s_embed_layer(s_formal_input)
            s_target_embeds = s_embed_layer(s_formal_target)

            t_input_embeds = t_embed_layer(t_formal_input)
            t_target_embeds = t_embed_layer(t_formal_target)

        s_index_embeds = torch.cat([s_input_embeds, s_target_embeds], dim=-1)
        t_index_embeds = torch.cat([t_input_embeds, t_target_embeds], dim=-1)

        s_index_embeds = s_index_embeds / s_index_embeds.std().clamp(min=1e-5)
        t_index_embeds = t_index_embeds / t_index_embeds.std().clamp(min=1e-5)

        # Hidden states
        s_hidden = student_outputs.last_hidden_state
        t_hidden = teacher_outputs.last_hidden_state

        s_hidden = s_hidden[:, :s_mask.size(1), :]
        t_hidden = t_hidden[:, :t_mask.size(1), :]

        # t_hidden_norm = t_hidden / t_hidden.std().clamp(min=1e-5)

        # Student query -> teacher index space
        s_q = self.dskd_index_projector_s2t(s_index_embeds).float()
        t_k = t_index_embeds.float()

        align = torch.matmul(s_q, t_k.transpose(-1, -2))
        align = align / math.sqrt(s_q.size(-1))

        align_mask = s_mask.float().unsqueeze(-1) * t_mask.float().unsqueeze(1)
        align = align.masked_fill(align_mask.eq(0), -1e4)

        # # ======================
        # # T2S: teacher -> student
        # # ======================
        # t2s_weight = torch.softmax(align, dim=-1)

        # t_value_for_student = self.dskd_value_projector_t2s(
        #     t_target_embeds + t_hidden_norm
        # ).float()

        # t2s_hidden = torch.matmul(t2s_weight, t_value_for_student)
        # t2s_logits = self.student.model.model.lm_head(t2s_hidden)

        # t2s_loss = self.masked_kl_loss(
        #     student_logits=student_outputs.logits[:, :s_mask.size(1), :],
        #     teacher_logits=t2s_logits.detach(),
        #     mask=s_mask,
        #     temperature=temperature,
        # )

        # ======================
        # S2T: student -> teacher
        # ======================
        s2t_weight = torch.softmax(align.transpose(-1, -2), dim=-1)

        s_value_for_teacher = self.dskd_value_projector_s2t(s_hidden).float()
        s2t_hidden = torch.matmul(s2t_weight, s_value_for_teacher)

        s2t_logits = self.teacher_lm_head(s2t_hidden)

        teacher_logits = self.teacher_lm_head(t_hidden)

        # s2t_loss = self.masked_kl_loss(
        #     student_logits=s2t_logits,
        #     teacher_logits=teacher_logits.detach(),
        #     mask=t_mask,
        #     temperature=temperature,
        # )
        s2t_loss = self.forward_kl(
            logits=s2t_logits,
            teacher_logits=teacher_logits.detach(),
            mask=t_mask
        )

        dskd_loss = s2t_loss

        return dskd_loss

    def knowledge_distillation_loss(
        self,
        student_outputs: StudentOutput,
        teacher_outputs: TeacherOutput = None,
        s_inputs=None,
        t_inputs=None,
        s_offsets_mapping=None,
        t_offsets_mapping=None,
        labels=None,
    ):
        kd_loss = 0
        temp_loss = torch.tensor(0)

        if teacher_outputs is not None:
            if teacher_outputs.hidden_states is not None:
                span_loss = 0
                der_loss = 0
                n_layer = teacher_outputs.hidden_states.size(0)
                span_weights = teacher_outputs.span_weights.squeeze(-1)
                _, B, N = span_weights.size()
                projectors = self.student.proj_hidden_layers

                mask = span_weights[-1].bool()  # [B, N]

                span_weights = span_weights ** self.args.p
                span_weights = span_weights / span_weights.sum(-1, keepdim=True)

                pair_weights = span_weights[-1].unsqueeze(2) * span_weights[-1].unsqueeze(1)
                mask = torch.eye(N, device=pair_weights.device).bool()  # (N, N)
                pair_weights[:, mask] = 0.0
                pair_weights = pair_weights / pair_weights.sum(dim=(1, 2), keepdim=True).clamp(min=1e-5)

                
                span_weights = span_weights.unsqueeze(-1)
                if self.args.span_loss:
                    for i in range(n_layer - 1, n_layer):
                        s_hidden = projectors[i](student_outputs.hidden_states[i])
                        t_didden = teacher_outputs.hidden_states[i]
                        span_w = span_weights[i]

                        state_loss = cosine_token_weight_loss(s_hidden, t_didden, span_w)
            
                        # span_loss += self.hidden_loss_weights[i] * state_loss
                        span_loss += state_loss

                        if torch.isnan(span_loss):
                            print('span_loss nan')
                if self.args.der_loss:
                    der_loss = derivative_loss(projectors[i](student_outputs.hidden_states),
                                            teacher_outputs.hidden_states,
                                            teacher_outputs.span_weights) / (n_layer - 1)

                    if torch.isnan(der_loss):
                        print('der_loss nan')

                kd_loss += 1.0 * span_loss
                kd_loss += 0.1 * der_loss
                # dskd_loss = self.dskd_with_cma(
                #     student_outputs=student_outputs,
                #     teacher_outputs=teacher_outputs,
                #     student_inputs=s_inputs,
                #     teacher_inputs=t_inputs,
                #     labels=labels,
                #     temperature=self.temperature,
                # )
                # kd_loss += 0.5 * dskd_loss
                


                s_hidden = F.normalize(student_outputs.embeddings, dim=-1, eps=1e-5)
                t_hidden = F.normalize(teacher_outputs.hidden_states[n_layer - 1], dim=-1, eps=1e-5)
                
                student_scores = torch.matmul(s_hidden, s_hidden.transpose(-1, -2))
                teacher_scores = torch.matmul(t_hidden, t_hidden.transpose(-1, -2))
                score_loss = F.mse_loss(student_scores, teacher_scores, reduction='none')
                score_loss = (score_loss * pair_weights).sum() / B
    
                kd_loss += self.args.geom_loss_weight * score_loss


                s_logits = self.student.model.model.lm_head(student_outputs.embeddings)
                t_logits = self.teacher_lm_head(teacher_outputs.hidden_states[n_layer - 1])
                
                s_map_logits = s_logits[:, :, self.s_id_mapping]
                t_map_logits = t_logits[:, :, self.t_id_mapping]
                kd_loss += self.soft_label_distill_loss(s_map_logits, t_map_logits)
                kd_loss += self.skewed_forward_kl(s_map_logits, t_map_logits)
                # mask=(t_map_logits.abs().sum(dim=-1) != 0)
                # kd_loss += self.forward_kl(s_map_logits, t_map_logits, mask)
                

        return kd_loss, temp_loss.item()

    
    def compute_loss(self, student_inputs, labels, teacher_inputs = None):
        t_offset_mapping = teacher_inputs.pop('offset_mapping', None)
        teacher_outputs = self.get_teacher_eval(teacher_inputs)
        
        s_offset_mapping = student_inputs.pop('offset_mapping', None)
        student_outputs = self.student.decode(student_inputs)
        
        hard_loss = self.student_loss_function(student_outputs.logits, 
                                               labels.view(-1), self.s_vocab_size)

        kd_loss, _t_loss_= 0, 0

        if self.args.knowledge_distillation and teacher_outputs is not None:
            kd_loss, _t_loss_ = self.knowledge_distillation_loss(
                student_outputs,
                teacher_outputs,
                student_inputs,
                teacher_inputs,
                s_offset_mapping,
                t_offset_mapping,
                labels=labels,
            )
        loss = self.alpha * hard_loss + (1.0 - self.alpha) * kd_loss

        self.step += 1

        return loss, hard_loss
    

def train(args: Arguments, trainer: Trainer, evaluator: Evaluator, grad_accum_steps=1):
    trainer.student.train()
    trainer.student.model.train()

    train_loader = trainer.train_loader

    optimizer = optim.AdamW(trainer.student.model.parameters(), lr=args.learning_rate)
    optimizer.add_param_group({"params": trainer.student.proj_hidden_layers.parameters(), "lr": 5e-4, "weight_decay": 0.0})
    optimizer = optim.AdamW(
        trainer.student.model.parameters(),
        lr=args.learning_rate
    )

    # hidden projectors
    optimizer.add_param_group({
        "params": trainer.student.proj_hidden_layers.parameters(),
        "lr": 5e-4,
        "weight_decay": 0.0
    })

    # ===== DSKD projectors =====

    optimizer.add_param_group({
        "params": trainer.dskd_index_projector_s2t.parameters(),
        "lr": 5e-4,
        "weight_decay": 0.0
    })

    optimizer.add_param_group({
        "params": trainer.dskd_value_projector_t2s.parameters(),
        "lr": 5e-4,
        "weight_decay": 0.0
    })

    optimizer.add_param_group({
        "params": trainer.dskd_value_projector_s2t.parameters(),
        "lr": 5e-4,
        "weight_decay": 0.0
    })
    num_steps = len(train_loader) // grad_accum_steps + 1
    total_traning_steps = num_steps * args.num_train_epochs

    scaler = GradScaler()

    scheduler = get_scheduler(
        name='cosine_with_min_lr',
        optimizer=optimizer,
        num_warmup_steps=int(total_traning_steps * args.warmup_ratio),
        # num_warmup_steps=0,
        num_training_steps=total_traning_steps,
        scheduler_specific_kwargs={'min_lr': 5e-6}
    )

    best_result = 0

    # Training loop
    for epoch in range(args.num_train_epochs):
        print(('\n' + '%8s' + '%14s' + '%17s' * 2) % ('epoch', 'memory', 'loss', 'student_loss'))
        p_bar = tqdm(train_loader, total=len(train_loader))
        loss_total = 0
        student_loss_total = 0
        step = 0

        for batch in p_bar:
            student_inputs, teacher_inputs, labels = batch

            labels = labels.to(trainer.student.device)
            with autocast(dtype=torch.bfloat16):
                loss, student_loss = trainer.compute_loss(student_inputs, labels, teacher_inputs)

            scaler.scale(loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(trainer.student.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
        
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_total += loss.item()
            student_loss_total += student_loss.item()
            step += 1

            memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
            s = ('%8s' + '%14s' + '%17.5g' * 2) % (f'{epoch + 1}/{args.num_train_epochs}', memory,
                                                    loss_total / step, student_loss_total / step)
            p_bar.set_description(s)

            if torch.isnan(loss):
                break

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            evaluator.model = trainer.student.model.model 
            dolly = evaluator.evaluate_benchmark_dataset(
                dataset_path=args.val_data,
                dataset_name='dolly', batch_size=64,
                max_seq_length=256, max_new_tokens=512)
        if dolly > best_result:
            best_result = dolly
            trainer.student.save(args.output_dir)
            
        trainer.student.save(args.output_dir + f'-epoch{epoch}')
            
        

    

