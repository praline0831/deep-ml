import torch
import torch.nn.functional as F

def distillation_loss(
	student_logits: torch.Tensor,
	teacher_logits: torch.Tensor,
	temperature: float = 1.0
) -> torch.Tensor:
	"""
	Compute knowledge distillation loss using PyTorch.
	
	Args:
		student_logits: Logits from student model
		teacher_logits: Logits from teacher model
		temperature: Softmax temperature
		
	Returns:
		Distillation loss
	"""
	# Your code here
	student_logits = F.log_softmax(student_logits / temperature, dim=-1)
	teacher_logits = F.softmax(teacher_logits / temperature, dim=-1)
	# reshape 到至少 2D，确保 batchmean 只除以 batch 数
	if student_logits.dim() == 1:
		student_logits = student_logits.unsqueeze(0)
		teacher_logits = teacher_logits.unsqueeze(0)
	return F.kl_div(student_logits, teacher_logits, reduction="batchmean") * (temperature **2)