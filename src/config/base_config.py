"""Base configuration for PredictionRTS"""

# Map and data identifiers
map_id = "MarineMicro_MvsM_4"
data_id = "6"

# Model parameters
N = 10
K = 5

# Model configuration
model_conf = "DecisionTransformer"

# Training parameters
params = {
    "mdl_spatial_prior": False,
    "mdl_init_embedding_freeze": False,
    "mdl_init_embedding_train": False,
    "mdl_attn_sim_bias": False,
    "N": N,
    "K": K,
}

# Model name based on configuration
if model_conf == "DecisionTransformer":
    model_name = "decisionTransformer"
else:
    model_name = "sgTransformer" if params["mdl_attn_sim_bias"] else "trajTransformer"

# Generate suffix
suffix = ""
if params.get("mdl_spatial_prior"):
    suffix += "_sp"
if params.get("mdl_init_embedding_freeze"):
    suffix += "_embedF"
if params.get("mdl_init_embedding_train"):
    suffix += "_embedT"
