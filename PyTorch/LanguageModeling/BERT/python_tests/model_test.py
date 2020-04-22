import torch
from torch.cuda import nvtx

device = torch.device("cuda")

# N is batch size; D_in is hidden_size in BERT;
# H is all_head_size_in_bert; D_out is output dimension.
N, D_in, H, D_out = 4, 1024, 4096, 4096

# Create random Tensors to hold inputs and outputs
x = torch.randn(N, 384, D_in, dtype=torch.float16, device = device)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H)
    )

model.to(device)
model.eval()
#print(model)
model.half()

for t in range(1):
    nvtx.range_push("start")
    y_pred = model(x)
    #print(y_pred)
    nvtx.range_pop()

#nvtx.range_push("start")
#y_pred =model(x);
#nvtx.range_pop;
