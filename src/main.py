import torch
from train import MyLanguageModel, ConfigModelConfig, generate_text


# Initialize the model
config = ConfigModelConfig()
model = MyLanguageModel(config)

# Load the checkpoint
checkpoint = torch.load("best_model.pt", map_location=config.device)  # or map_location=config.device

# Extract model weights
model_weights = checkpoint['model_state_dict']


# Load the weights into the model
model.load_state_dict(model_weights)
model.to(config.device)
model.eval()  # Set model to evaluation mode

# Generate text
prompt = " "
generated_text = generate_text(prompt, max_length=config.max_seq_len)

print(generated_text)