from ultralytics import YOLOv10
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        default="yolov10s",
                        help="Model name to export to onnx",
                        choices=["yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x"])

    parser.add_argument("--imgsz", type=int, nargs=2,
                        default=(480, 640),
                        help="Image size for the model")
    args = parser.parse_args()

    print("LOADING MODEL")
    model = YOLOv10.from_pretrained(f"jameslahm/{args.model}")

    print("EXPORTING MODEL TO ONNX")
    model.export(format='onnx', simplify=False, imgsz=args.imgsz)

    print("FORWARD PASS")
    results = model.predict(source="./sample-image.png",  conf=0.5)
    print("END")
