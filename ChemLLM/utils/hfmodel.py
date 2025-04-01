from molfeat.trans.pretrained.hf_transformers import HFExperiment
from molfeat.trans.pretrained.hf_transformers import HFModel
from molfeat.store import ModelInfo
from molfeat.store import ModelStore



molt5_card = ModelInfo(
    name = "MolT5",
    inputs = "smiles",
    type="pretrained",
    group="huggingface",
    version=0,
    submitter="Desmond Gilmour",
    description="MolT5 is a self-supervised learning framework that pretrains transformer-based models on vast amounts of unlabeled natural language text and molecule strings allowing generation of high-quality outputs for molecule captioning and text-based molecule generation.",
    representation="line-notation",
    require_3D=False,
    tags = ["smiles", 'huggingface', "transformers", "text2text", "T5", "Zinc-15", "ChEBI-20"],
    authors= ["Tuan Manh Lai", "Carl Edwards", "Kevin Ros", "Garret Honke", "Kyunghyun Cho", "Heng Ji"],
    reference = "https://arxiv.org/pdf/2204.11817.pdf" 
)

# attempt to register the model
model = HFModel.register_pretrained("laituan245/molt5-large-smiles2caption", "laituan245/molt5-large-smiles2caption", molt5_card)