import torch
import torch.nn as nn

input = torch.Tensor([0.2, 7, 39])
target = torch.Tensor([1, 8, 37])

#Mean Squared Error Loss
mse_loss = nn.MSELoss()
mse_results = mse_loss(input, target)

print("Mean Squared Error Loss:", mse_results)

#Mean Absolute Error
mae_loss = torch.nn.L1Loss()
mae_results = mae_loss(input, target)

print("Mean Absolute Error:", mae_results)

#Cross Entropy Loss
cross_entropy_loss = nn.CrossEntropyLoss()
ce_results = cross_entropy_loss(input, target)

print("Cross Entropy Loss:", ce_results)

#Triple Margin Loss
triplet_margin_loss = nn.TripletMarginLoss(margin=1.0)
anchor = torch.Tensor([0.2, 7, 39])
triplet_result = triplet_margin_loss(anchor, input, target)

print("Triplet Margin Loss:", triplet_result)