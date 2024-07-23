from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Body
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path
from torch.utils.data import DataLoader
from anomalib.data import PredictDataset, Folder
from anomalib.engine import Engine
from anomalib.models import Padim
from anomalib import TaskType
import mimetypes
import time
from PIL import Image
import io
import base64
from torchvision import transforms

app = FastAPI(title="Anomalib API", description="API for anomaly detection using Anomalib", version="1.0.0")


class DatasetConfig(BaseModel):
    name: str = Field(..., title="Dataset Name", description="The name of the dataset.", example="custom_dataset")
    root: str = Field(..., title="Root Directory", description="The root directory of the dataset.", example="./datasets/custom")
    normal_train_dirs: List[str] = Field(..., title="Normal Train Directories", description="List of directories containing normal training images.", example=["train/normal"])
    abnormal_train_dirs: Optional[List[str]] = Field(None, title="Abnormal Train Directories", description="List of directories containing abnormal training images.", example=["train/abnormal"])
    mask_train_dirs: Optional[List[str]] = Field(None, title="Mask Train Directories", description="List of directories containing training masks.", example=["train/mask"])
    normal_eval_dirs: List[str] = Field(..., title="Normal Eval Directories", description="List of directories containing normal evaluation images.", example=["eval/normal"])
    abnormal_eval_dirs: List[str] = Field(..., title="Abnormal Eval Directories", description="List of directories containing abnormal evaluation images.", example=["eval/abnormal"])
    mask_eval_dirs: List[str] = Field(..., title="Mask Eval Directories", description="List of directories containing evaluation masks.", example=["eval/mask"])
    task: str = Field(..., title="Task Type", description="The type of task to perform (e.g., CLASSIFICATION, SEGMENTATION).", example="SEGMENTATION")
    image_size: List[int] = Field(..., title="Image Size", description="The size of the images.", example=[256, 256])


class ModelConfig(BaseModel):
    backbone: str = Field(..., title="Backbone", description="The backbone model to use.", example="resnet18")
    layers: List[str] = Field(..., title="Layers", description="List of layers to use.", example=["layer1", "layer2", "layer3"])


class CallbackConfig(BaseModel):
    class_path: str = Field(..., title="Callback Class Path", description="The class path of the callback.", example="lightning.pytorch.callbacks.EarlyStopping")
    init_args: Dict[str, Any] = Field(..., title="Initialization Arguments", description="Initialization arguments for the callback.", example={"monitor": "val_loss", "patience": 3})


class EngineConfig(BaseModel):
    callbacks: List[CallbackConfig] = Field(..., title="Callbacks", description="List of callbacks to use during training.")
    accelerator: str | None = Field(..., title="Accelerator", description="The type of accelerator to use (e.g., gpu, cpu).", example="gpu")
    devices: int | None = Field(..., title="Devices", description="The number of devices to use.", example=1)
    logger: bool = Field(..., title="Logger", description="Whether to use logging.", example=False)


class TrainConfig(BaseModel):
    max_epochs: int = Field(..., title="Max Epochs", description="The maximum number of epochs for training.", example=50)


class InferConfig(BaseModel):
    model_checkpoint: str = Field(..., title="Model Checkpoint", description="Path to the model checkpoint.", example="./checkpoints/model.ckpt")
    batch_size: int | None= Field(1, title="Batch Size", description="The batch size for inference.", example=32)


class Config(BaseModel):
    dataset: DatasetConfig
    model: ModelConfig
    engine: EngineConfig
    train: TrainConfig
    test: Dict[str, Any] = Field(..., title="Test Configuration", description="Configuration for testing.")
    benchmark: Dict[str, Any] = Field(..., title="Benchmark Configuration", description="Configuration for benchmarking.")
    infer: InferConfig


class FileMetadata(BaseModel):
    name: str = Field(..., title="File Name", description="The name of the file.")
    size: int = Field(..., title="File Size", description="The size of the file in bytes.")
    modified_time: str = Field(..., title="Modified Time", description="The last modified time of the file.", example="2023-06-26T12:00:00Z")
    created_time: str = Field(..., title="Created Time", description="The creation time of the file.", example="2023-06-26T10:00:00Z")
    path: str = Field(..., title="File Path", description="The path to the file.")
    mime_type: Optional[str] = Field(None, title="MIME Type", description="The MIME type of the file.", example="text/plain")
    custom_metadata: Optional[Dict[str, Any]] = Field(None, title="Custom Metadata", description="Any custom metadata associated with the file.", example={"author": "John Doe", "project": "AI Research"})


@app.post("/train")
def train_model(config: Config):
    """
    Train an anomaly detection model.

    This endpoint initializes a dataset, model, and engine based on the provided configuration,
    and starts the training process.

    Parameters:
    - config (Config): Configuration parameters for the training process.

    Returns:
    - dict: A message indicating the success of the training process.
    """
    try:
        dataset_config = config.dataset
        model_config = config.model
        engine_config = config.engine
        train_config = config.train

        # Initialize the data module
        datamodule = Folder(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            normal_dir=dataset_config.normal_train_dirs,
            abnormal_dir=dataset_config.abnormal_train_dirs,
            mask_dir=dataset_config.mask_train_dirs,
            task=TaskType[dataset_config.task],
            image_size=tuple(dataset_config.image_size),
        )

        datamodule.setup()

        # Initialize the model
        model = Padim(**model_config.dict())

        # Initialize the engine with callbacks
        engine_kwargs = engine_config.dict()
        engine_kwargs['callbacks'] = [eval(callback.class_path)(**callback.init_args) for callback in engine_config.callbacks]
        engine = Engine(**engine_kwargs)

        # Start training
        engine.fit(
            model=model,
            train_dataloaders=datamodule.train_dataloader(),
            val_dataloaders=datamodule.val_dataloader(),
            **train_config.dict()
        )

        return {"message": "Training completed successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test")
def test_model(config: Config):
    """
    Test a trained anomaly detection model.

    This endpoint loads a pre-trained model and tests it on the provided dataset configuration.

    Parameters:
    - config (Config): Configuration parameters for the testing process.

    Returns:
    - dict: Test results of the model.
    """
    try:
        dataset_config = config.dataset
        test_config = config.test

        # Initialize the data module
        datamodule = Folder(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            normal_dir=dataset_config.normal_eval_dirs,
            abnormal_dir=dataset_config.abnormal_eval_dirs,
            mask_dir=dataset_config.mask_eval_dirs,
            task=TaskType[dataset_config.task],
            image_size=tuple(dataset_config.image_size),
        )

        datamodule.setup()

        # Initialize the engine with configuration
        engine_kwargs = config.engine.dict()
        engine_kwargs['callbacks'] = [eval(callback.class_path)(**callback.init_args) for callback in config.engine.callbacks]
        engine = Engine(**engine_kwargs)

        # Load the model
        model = Padim.load_from_checkpoint(test_config['model_checkpoint'])

        # Test the model
        test_results = engine.test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            **test_config
        )

        return {"test_results": test_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


'''@app.post("/benchmark")
def benchmark_model(config: Config):
    """
    Benchmark a trained anomaly detection model.

    This endpoint loads a pre-trained model and benchmarks it on the provided dataset configuration.

    Parameters:
    - config (Config): Configuration parameters for the benchmarking process.

    Returns:
    - dict: Benchmark results of the model.
    """
    try:
        dataset_config = config.dataset
        benchmark_config = config.benchmark

        # Initialize the data module
        datamodule = Folder(
            name=dataset_config.name,
            root=Path(dataset_config.root),
            normal_dir=dataset_config.normal_eval_dirs,
            abnormal_dir=dataset_config.abnormal_eval_dirs,
            mask_dir=dataset_config.mask_eval_dirs,
            task=TaskType[dataset_config.task],
            image_size=tuple(dataset_config.image_size),
        )

        datamodule.setup()

        # Initialize the engine with configuration
        engine_kwargs = config.engine.dict()
        engine_kwargs['callbacks'] = [eval(callback.class_path)(**callback.init_args) for callback in config.engine.callbacks]
        engine = Engine(**engine_kwargs)

        # Load the model
        model = Padim.load_from_checkpoint(benchmark_config['model_checkpoint'])

        # Benchmark the model
        benchmark_results = engine.test(
            model=model,
            dataloaders=datamodule.test_dataloader(),
            **benchmark_config
        )

        return {"benchmark_results": benchmark_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))'''


@app.post("/infer")
async def infer_image(
        infer_config: str = Form('{\"model_checkpoint\": \"C:/Users/Golden Bit/Desktop/projects_in_progress/protom/anomaly_detection-protom_project/models/model.ckpt\", \"batch_size\": 32}', description="Configuration parameters for the inference process.", example='{\"model_checkpoint\": \"C:/Users/Golden Bit/Desktop/projects_in_progress/protom/anomaly_detection-protom_project/models/model.ckpt\", \"batch_size\": 32}'),
        engine_config: str = Form('{\"callbacks\":[],\"accelerator\":\"gpu\",\"devices\":1\"logger\":false}', description="Configuration parameters for the engine", example='{\"callbacks\":[{\"class_path\":\"lightning.pytorch.callbacks.EarlyStopping\",\"init_args\":{\"monitor\":\"val_loss\",\"patience\":3}}],\"accelerator\":\"gpu\",\"devices\":1,\"logger\":false}'),
        file: Optional[UploadFile] = File(None, description="The image file to upload for inference."),
        base64_image: Optional[str] = Form(None, description="The base64 encoded image string for inference."),
        save_dir: str = Form(..., description="The directory to save the uploaded image."),
        is_persistent: bool = Form(False, description="Flag to indicate if the image should be saved permanently."),
        custom_filename: Optional[str] = Form(None, description="Custom filename to save the image.")
):
    """
    Perform inference using a trained anomaly detection model on an uploaded image or a base64 encoded image.

    This endpoint loads a pre-trained model and performs inference on the provided image.

    Parameters:
    - infer_config (str): Configuration parameters for the inference process in JSON format. Example: '{"model_checkpoint": "C:/Users/Golden Bit/Desktop/projects_in_progress/protom/anomaly_detection-protom_project/models/model.ckpt", "batch_size": 32}'
    - engine_config (str): Configuration parameters for the engine in JSON format. Example: '{"callbacks":[{"class_path":"lightning.pytorch.callbacks.EarlyStopping","init_args":{"monitor":"val_loss","patience":3}}],"accelerator":"gpu","devices":1,"logger":false}'
    - file (UploadFile, optional): The image file to upload for inference.
    - base64_image (str, optional): The base64 encoded image string for inference.
    - save_dir (str): The directory to save the uploaded image.
    - is_persistent (bool): Flag to indicate if the image should be saved permanently.
    - custom_filename (str, optional): Custom filename to save the image.

    Returns:
    - dict: Inference results of the model.
    """
    if True: #try:
        # Parse JSON strings to dictionaries
        infer_config_dict = json.loads(infer_config)
        engine_config_dict = json.loads(engine_config)

        # Validate and create Pydantic models
        infer_config_obj = InferConfig(**infer_config_dict)
        engine_config_obj = EngineConfig(**engine_config_dict)

        if not file and not base64_image:
            raise HTTPException(status_code=400, detail="Either 'file' or 'base64_image' must be provided")

        # Load the image from file or base64 string and save it
        if file:
            content = await file.read()
            image = Image.open(io.BytesIO(content)).convert("RGB")
            filename = custom_filename if custom_filename else file.filename
        else:
            image_data = base64.b64decode(base64_image)
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            filename = custom_filename if custom_filename else "uploaded_image.png"

        file_path = Path(save_dir) / filename

        if is_persistent and file_path.exists():
            raise HTTPException(status_code=400, detail=f"File '{filename}' already exists")

        file_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(file_path)

        # Create inference dataset
        inference_dataset = PredictDataset(path=file_path.parent)
        inference_dataloader = DataLoader(dataset=inference_dataset, batch_size=infer_config_obj.batch_size)

        # Initialize the engine with configuration
        engine_kwargs = engine_config_obj.dict()
        engine_kwargs['callbacks'] = [eval(callback.class_path)(**callback.init_args) for callback in engine_config_obj.callbacks]
        engine = Engine(**engine_kwargs)

        # Load the model
        model = Padim.load_from_checkpoint(infer_config_obj.model_checkpoint)

        # Perform inference
        #predictions = engine.predict(
        #    model=model,
        #    #dataloaders=inference_dataloader,
        #    data_path=file_path,
        #    #return_predictions=True
        #)

        predictions = engine.test(
            model=model,
            dataloaders=inference_dataloader,
            #data_path=file_path,
            #return_predictions=True
        )

        if not is_persistent:
            file_path.unlink()  # Delete the file after inference

        return {"inference_results": predictions}
    #except Exception as e:
    #    raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload", response_model=FileMetadata)
async def upload_file(
        file: UploadFile = File(..., description="The file to upload."),
        subdir: Optional[str] = Form(None, description="The subdirectory to save the file in."),
        file_description: Optional[str] = Form(None, description="Custom description for the file."),
        extra_metadata: Optional[str] = Form(None, description="Extra metadata for the file in JSON format.")
):
    """
    Upload a file to the server.

    This endpoint allows users to upload a file to a specified subdirectory on the server,
    with optional custom metadata.

    Parameters:
    - file (UploadFile): The file to upload.
    - subdir (str, optional): The subdirectory to save the file in.
    - file_description (str, optional): Custom description for the file.
    - extra_metadata (str, optional): Extra metadata for the file in JSON format.

    Returns:
    - FileMetadata: Metadata of the uploaded file.
    """
    try:
        if subdir:
            file_path = os.path.join(subdir, file.filename)
        else:
            file_path = file.filename

        file_path = file_path.replace("\\", "/")

        content = await file.read()
        custom_metadata = {"file_description": file_description} if file_description else {}
        if extra_metadata:
            custom_metadata.update(json.loads(extra_metadata))

        save_file(file_path, content, custom_metadata)
        metadata = get_file_metadata(file_path)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload/multiple", response_model=List[FileMetadata])
async def upload_multiple_files(
        files: List[UploadFile] = File(..., description="The list of files to upload."),
        subdir: Optional[str] = Form(None, description="The subdirectory to save the files in."),
        file_description: Optional[str] = Form(None, description="Custom description for the files."),
        extra_metadata: Optional[str] = Form(None, description="Extra metadata for the files in JSON format.")
):
    """
    Upload multiple files to the server.

    This endpoint allows users to upload multiple files to a specified subdirectory on the server,
    with optional custom metadata for each file.

    Parameters:
    - files (List[UploadFile]): The list of files to upload.
    - subdir (str, optional): The subdirectory to save the files in.
    - file_description (str, optional): Custom description for the files.
    - extra_metadata (str, optional): Extra metadata for the files in JSON format.

    Returns:
    - List[FileMetadata]: List of metadata for the uploaded files.
    """
    try:
        metadata_list = []
        for file in files:
            if subdir:
                file_path = os.path.join(subdir, file.filename)
            else:
                file_path = file.filename

            file_path = file_path.replace("\\", "/")

            content = await file.read()
            custom_metadata = {"file_description": file_description} if file_description else {}
            if extra_metadata:
                custom_metadata.update(json.loads(extra_metadata))

            save_file(file_path, content, custom_metadata)
            metadata = get_file_metadata(file_path)
            metadata_list.append(metadata)

        return metadata_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/update/{file_path:path}", response_model=FileMetadata)
async def update_file(
        file_path: str,
        file: UploadFile = File(..., description="The new file content."),
        file_description: Optional[str] = Form(None, description="Custom description for the file."),
        extra_metadata: Optional[str] = Form(None, description="Extra metadata for the file in JSON format.")
):
    """
    Update an existing file on the server.

    This endpoint allows users to update an existing file with new content and optional custom metadata.

    Parameters:
    - file_path (str): The path of the file to update.
    - file (UploadFile): The new file content.
    - file_description (str, optional): Custom description for the file.
    - extra_metadata (str, optional): Extra metadata for the file in JSON format.

    Returns:
    - FileMetadata: Metadata of the updated file.
    """
    try:
        content = await file.read()
        custom_metadata = {"file_description": file_description} if file_description else {}
        if extra_metadata:
            custom_metadata.update(json.loads(extra_metadata))

        save_file(file_path, content, custom_metadata)
        metadata = get_file_metadata(file_path)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete", response_model=Dict[str, Any])
async def delete_files(
        file_paths: Optional[List[str]] = Query(None, description="List of file paths to delete."),
        directory: Optional[str] = Query(None, description="Directory to delete all files from.")
):
    """
    Delete files from the server.

    This endpoint allows users to delete specific files by listing their paths,
    or to delete all files in a specified directory.

    Parameters:
    - file_paths (List[str], optional): List of file paths to delete.
    - directory (str, optional): Directory to delete all files from.

    Returns:
    - dict: A message indicating the success of the deletion process.
    """
    try:
        if file_paths:
            for file_path in file_paths:
                file_path = Path(file_path)
                if file_path.exists():
                    file_path.unlink()
                    metadata_path = file_path.with_suffix(file_path.suffix + ".json")
                    if metadata_path.exists():
                        metadata_path.unlink()
                else:
                    raise HTTPException(status_code=404, detail=f"File '{file_path}' not found")

        if directory:
            directory_path = Path(directory)
            if directory_path.exists() and directory_path.is_dir():
                for item in directory_path.iterdir():
                    if item.is_file():
                        item.unlink()
                        metadata_path = item.with_suffix(item.suffix + ".json")
                        if metadata_path.exists():
                            metadata_path.unlink()
                    else:
                        raise HTTPException(status_code=400, detail=f"Path '{item}' is not a file")
            else:
                raise HTTPException(status_code=404, detail=f"Directory '{directory}' not found")

        return {"message": "Files deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def save_file(file_path: str, content: bytes, custom_metadata: Dict[str, Any]):
    """
    Save a file to the specified path with custom metadata.

    Parameters:
    - file_path (str): The path to save the file.
    - content (bytes): The content of the file.
    - custom_metadata (Dict[str, Any]): Custom metadata to associate with the file.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)
    save_file_metadata(file_path, custom_metadata)


def get_file_metadata(file_path: str) -> FileMetadata:
    """
    Retrieve metadata of a specified file.

    Parameters:
    - file_path (str): The path of the file.

    Returns:
    - FileMetadata: Metadata of the specified file.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File '{file_path}' not found")

    stat = file_path.stat()
    mime_type, _ = mimetypes.guess_type(file_path)
    metadata = {
        "name": file_path.name,
        "size": stat.st_size,
        "modified_time": time.ctime(stat.st_mtime),
        "created_time": time.ctime(stat.st_ctime),
        "path": str(file_path),
        "mime_type": mime_type,
        "custom_metadata": load_file_metadata(file_path)
    }
    return FileMetadata(**metadata)


def save_file_metadata(file_path: Path, custom_metadata: Dict[str, Any]):
    """
    Save custom metadata for a specified file.

    Parameters:
    - file_path (Path): The path of the file.
    - custom_metadata (Dict[str, Any]): Custom metadata to save.
    """
    metadata_path = file_path.with_suffix(file_path.suffix + ".json")
    with open(metadata_path, "w") as f:
        json.dump(custom_metadata, f)


def load_file_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Load custom metadata of a specified file.

    Parameters:
    - file_path (Path): The path of the file.

    Returns:
    - Dict[str, Any]: Custom metadata of the file.
    """
    metadata_path = file_path.with_suffix(file_path.suffix + ".json")
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
