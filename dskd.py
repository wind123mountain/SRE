import math
import torch
import torch.nn.functional as F


def align_sequences(tea_seq, stu_seq, student_tokenizer, teacher_tokenizer):
    i, j = 0, 0
    t2s_align, s2t_align = [], []
    history_tea_seq, history_stu_seq = "", ""
    
    tea_seq = [token.replace('▁', '').replace('Ġ', '') for token in tea_seq]
    stu_seq = [token.replace('▁', '').replace('Ġ', '') for token in stu_seq]

    while i < len(tea_seq) and j < len(stu_seq):
        if history_tea_seq == history_stu_seq and (
            tea_seq[i] == stu_seq[j] or (
                tea_seq[i] == teacher_tokenizer.eos_token and \
                stu_seq[j] == student_tokenizer.eos_token
            )
        ):
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            t2s_align.append(i)
            s2t_align.append(j)
            i += 1
            j += 1
        elif len(history_tea_seq) > len(history_stu_seq):
            history_stu_seq += stu_seq[j]
            j += 1
        elif len(history_tea_seq) < len(history_stu_seq):
            history_tea_seq += tea_seq[i]
            i += 1
        else:
            history_tea_seq += tea_seq[i]
            history_stu_seq += stu_seq[j]
            i += 1
            j += 1
            
    return t2s_align, s2t_align


class VariousDivergence():
    def __init__(self, args, padding_id=-100) -> None:
        super(VariousDivergence, self).__init__(args, padding_id=padding_id)
        self.kd_rate = args.kd_rate
        self.kd_temp = args.kd_temperature
        self.tea_temp = args.teacher_temperature
        self.kd_objective = args.kd_objective
        self.args = args

        if self.kd_objective == "forward_kl":
            self.dist_func = self.compute_forward_kl_divergence
        elif self.kd_objective == "reverse_kl":
            self.dist_func = self.compute_reverse_kl_divergence
        elif self.kd_objective == "adaptive_kl":
            self.dist_func = self.compute_adaptive_kl_divergence
        elif self.kd_objective == "skewed_forward_kl":
            self.dist_func = self.compute_skewed_forward_kl_divergence
        elif self.kd_objective == "skewed_reverse_kl":
            self.dist_func = self.compute_skewed_reverse_kl_divergence
        elif self.kd_objective == "js_divergence":
            self.dist_func = self.compute_js_divergence
        else:
            raise NameError(f"Unsupported kd_objective for `{self.kd_objective}'")

    def compute_forward_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["forward_kl"] = kld

        return kld
    
    def compute_reverse_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        probs = torch.softmax(logits, -1, dtype=torch.float32)
        lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (probs * (lprobs - teacher_lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)

        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["reverse_kl"] = kld

        return kld
    
    def compute_skewed_forward_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = self.args.skew_lambda * teacher_probs + (1 - self.args.skew_lambda) * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        teacher_lprobs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
        kld = (teacher_probs * (teacher_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_forward_kl"] = kld

        return kld
    
    def compute_skewed_reverse_kl_divergence(
        self, 
        logits, 
        teacher_logits, 
        target, 
        reduction="sum", 
        log=None, 
        use_tea_temp=False
    ):
        logits = logits / self.kd_temp
        teacher_logits = teacher_logits / self.kd_temp
        teacher_logits = teacher_logits / self.tea_temp if use_tea_temp else teacher_logits

        student_probs = torch.softmax(logits, -1, dtype=torch.float32)
        teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
        mixed_probs = (1 - self.args.skew_lambda) * teacher_probs + self.args.skew_lambda * student_probs
        mixed_lprobs = torch.log(mixed_probs)
        student_lprobs = torch.log_softmax(logits, -1, dtype=torch.float32)
        # teacher_lprobs = torch.log_softmax(teacher_logits / self.tea_temp / self.kd_temp, -1, dtype=torch.float32)
        kld = (student_probs * (student_lprobs - mixed_lprobs))
        inf_mask = logits.isinf() | logits.isnan() | teacher_logits.isinf() | teacher_logits.isnan()
        kld = kld.masked_fill_(inf_mask, 0.0).sum(-1)
        
        if reduction == "sum":
            pad_mask = target.eq(self.padding_id)
            kld = kld.masked_fill_(pad_mask, 0.0)
            kld = kld.sum()

            if log is not None:
                log["skewed_reverse_kl"] = kld

        return kld

class DualSpaceKDV2WithETA(VariousDivergence):
    def __init__(self, args, padding_id=-100) -> None:
        super().__init__(args, padding_id=padding_id)

    def forward(
        self, 
        distiller, 
        batch, 
        logging_output
    ):
        model = distiller.student_model
        teacher_model = distiller.teacher_model
        teacher_model.eval()

        self.distiller = distiller
        outputs = model(**batch["input_batch"], output_hidden_states=True)
        logits = outputs.logits
        log = {}

        with torch.no_grad():
                teacher_outputs = teacher_model(
                    **batch["teacher_input_batch"], 
                    output_hidden_states=True
                )
        kd_loss = self.compute_dual_space_kd_loss_with_cma(
            outputs, teacher_outputs, batch, distiller, log
        )
        

    def compute_dual_space_kd_loss_with_cma(
        self, outputs, teacher_outputs, batch, distiller
    ):
        target = batch["label_batch"]["label"]
        teacher_target = batch["teacher_label_batch"]["label"]
          
        pad_mask = target.ne(self.padding_id)
        teacher_pad_mask = teacher_target.ne(self.padding_id)

        hiddens = outputs.hidden_states[-1]
        teacher_hiddens = teacher_outputs.hidden_states[-1]

        if hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.embed_tokens
        elif hasattr(distiller.student_model, "model") \
            and hasattr(distiller.student_model.model, "model") \
            and hasattr(distiller.student_model.model.model, "embed_tokens"):
            stu_embed_tokens = distiller.student_model.model.model.embed_tokens
        elif hasattr(distiller.student_model, "transformer") \
            and hasattr(distiller.student_model.transformer, "wte"):
            stu_embed_tokens = distiller.student_model.transformer.wte
        else:
            raise NotImplementedError

        if hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "model") \
            and hasattr(distiller.teacher_model.model, "model") \
            and hasattr(distiller.teacher_model.model.model, "embed_tokens"):
            tea_embed_tokens = distiller.teacher_model.model.model.embed_tokens
        elif hasattr(distiller.teacher_model, "transformer") \
            and hasattr(distiller.teacher_model.model, "wte"):
            tea_embed_tokens = distiller.teacher_model.transformer.wte
        else:
            raise NotImplementedError

        formal_input = torch.where(pad_mask, batch["input_batch"]["input_ids"], torch.zeros_like(target))
        formal_target = torch.where(pad_mask, target, torch.zeros_like(target))
        stu_input_embeds = stu_embed_tokens(formal_input).detach()
        stu_target_embeds = stu_embed_tokens(formal_target).detach()

        formal_teacher_input = torch.where(teacher_pad_mask, batch["teacher_input_batch"][f"input_ids"], torch.zeros_like(teacher_target))
        formal_teacher_target_for_index = torch.where(teacher_pad_mask, teacher_target, torch.zeros_like(teacher_target))

        tea_input_embeds = tea_embed_tokens(formal_teacher_input).detach()
        tea_target_embeds = tea_embed_tokens(formal_teacher_target_for_index).detach()

        stu_index_embeds = torch.cat([stu_input_embeds, stu_target_embeds], -1)
        tea_index_embeds = torch.cat([tea_input_embeds, tea_target_embeds], -1)

        norm_tea_index_embeds = tea_index_embeds / tea_index_embeds.std()

        stu_q_hiddens = distiller.query_projector(stu_index_embeds).float()
        tea_k_hiddens = norm_tea_index_embeds.float()

        # teacher space
        if distiller.part_teacher_head_pinv is not None:
            stu_lmhead = distiller.student_model.lm_head.weight.detach().transpose(0, 1)
            stu_lmhead = stu_lmhead[:, distiller.student_overlap_token_ids]
            s2t_proj = stu_lmhead @ distiller.part_teacher_head_pinv
            stu_v_hiddens = hiddens @ s2t_proj
        else:
            stu_v_hiddens = distiller.s2t_projectors(hiddens).float()  # n x d x d x D -> n x D

      
        align = stu_q_hiddens.matmul(tea_k_hiddens.transpose(-1, -2))
        align = align / math.sqrt(2 * teacher_hiddens.shape[-1])
        align_mask = pad_mask.float().unsqueeze(-1) * teacher_pad_mask.float().unsqueeze(1)
        align = align + (1.0 - align_mask) * (-100000)

        # teacher space
        s2t_weight = torch.softmax(align.transpose(-1, -2), -1).to(hiddens)
        s2t_hiddens = s2t_weight.matmul(stu_v_hiddens)  # m x n x n x D -> m x D
        s2t_logits = distiller.teacher_model.lm_head(s2t_hiddens)
        s2t_kd_loss = self.dist_func(
            s2t_logits, teacher_outputs.logits, teacher_target, reduction="none"
        )
        s2t_kd_loss = (s2t_kd_loss * teacher_pad_mask).sum() / batch["label_batch"]["loss_denom"]
        
        return s2t_kd_loss
      
