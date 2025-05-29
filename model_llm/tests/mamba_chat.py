import os
import sys
ROOT = os.getcwd()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT+"/model_llm"))    

import torch
# from mamba_py.mamba_lm import from_pretrained
from transformers import AutoTokenizer,MambaForCausalLM
import time 
# from mamba_py.modeling_mamba import MambaForCausalLM
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-2.8b-hf")
# model = MambaForCausalLM.from_pretrained("state-spaces/mamba-2.8b-hf")
# input_ids = tokenizer("Mamba is a type of", return_tensors="pt")["input_ids"]

# out = model.generate(input_ids, max_new_tokens=10)
# print(tokenizer.batch_decode(out))


# Load model directly
model_torch = MambaForCausalLM.from_pretrained('state-spaces/mamba-2.8b-hf').to(device)
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')

# input_text = "Mamba is a type of" 
input_text = """Most state-of-the-art techniques for quantizing LLMs are
based on the empirical observation of outlier channels (Bon-
darenko et al., 2021), a small percentage of model dimen-
sions with a dynamic range that is consistently larger than
the rest. This phenomenon complicates activation quan-
tization since the large abs max values from the outlier
channels deteriorate the effective bit precision of the remain-
ing channels. A possible solution would be maintaining a
different quantization scale for each channel, which is not
hardware-friendly on current GPU architectures (Xiao et al.,
2024). Various strategies have been proposed to circum-
vent this issue. For instance, some methods treat outlier
channels separately, either by maintaining them in floating
point format (Dettmers et al., 2022) or by representing them
with two integer channels each (Zhang et al., 2024). Other
approaches modify the transformer architecture to prevent
the emergence of outliers (Bondarenko et al., 2023), while
some partially shift the quantization difficulty to the weights,
thereby mitigating the impact of outliers (Xiao et al., 2024).
We make the first steps towards post-training quantization
for recurrent LLMs, focusing on the Mamba (Gu & Dao,
2023) model family. We analyse the activation patterns of
Mamba to assess the presence of outliers, which we define
as those channels having an absolute maximum activation
beyond six standard deviations from the layer mean, fol-
lowing prior practice (Bondarenko et al., 2021). Figure 1
reports the pre-activations of the linear block of a layer from
Mamba-130m (similar results were observed for the other
model sizes), measured running the model on a subset of
WikiText-2 (Merity et al., 2016). We observe distinct out-
lier patterns. The pre-activations of the three largest linear
layers (in, x, and out), consistently with what was observed
for attention-based LLMs, show outliers accounting for less
than 1% of channels. However, while the outliers of the first
linear block are mostly consistent across layers, the remain-
ing two blocks exhibit no regular behavior. The linear layer
projecting the SSM’s time steps (dt) shows almost no out-
liers. Similarly to (Dettmers et al., 2022), we further assess
the importance of the outlier channels for the model’s pre-
dictions by evaluating the impact of zeroing out the outliers
on downstream accuracy. For Mamba-130m and Mamba-
2.8b, we observe a drop in average accuracy of 12.61% and
17.49%, respectively, suggesting that these channels play a
significant role in the model dynamics. Extended results are
available in the Appendix in Table 2.
"""
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

from ptflops import get_model_complexity_info
input_ids = input_ids[:,:512]
# time1 = time.time()
# with torch.no_grad():
#     model_torch(input_ids)
# time2 = time.time()
# with torch.no_grad():
#     model_torch(input_ids[:,:1])
# time3 = time.time()    
# print(time2-time1,time3-time2)
input_shape = (512,)
with torch.no_grad():
    macs, params = get_model_complexity_info(model_torch,
                                            input_shape,
                                            as_strings=False, 
                                            print_per_layer_stat=True,
                                            input_constructor=lambda input_res: torch.randint(0, 255, (1, *input_res), dtype=torch.int32).to(device)
                                            )
print(f"MACs: {macs}")
print(f"Parameters: {params}")

# output_ids = model_torch.generate(input_ids)
# output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
# print(output_text)