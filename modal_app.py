import modal
from src.fruitless_direction.steering import load_model, safe_generate

gpu = modal.gpu.A100(memory=40)  # cheap burst; tweak as needed
volume = modal.Volume.from_local_dir(
    ".", 
    mount_path="/modal/fruitless-direction"
)

stub = modal.Stub(
    "fruitless-direction",
    image=modal.Image.debian_slim()
        .pip_install_from_pyproject("pyproject.toml"),
    volumes={"/modal/fruitless-direction": volume},
)

@stub.function(gpu=gpu, timeout=600)
def chat(prompt: str) -> str:
    pipe = load_model()
    return safe_generate(pipe, prompt)
