import argparse
from .inference import predict_image


def main():
    parser = argparse.ArgumentParser(
        description="Predict forest/tree class for an image using the fastai model."
    )
    parser.add_argument("image_path", help="Path to the image file (jpg/png).")
    args = parser.parse_args()

    result = predict_image(args.image_path)

    print(f"\nPredicted label: {result['label']}\n")
    print("Class probabilities:")
    for label, p in result["probs"].items():
        print(f"  {label:20s} {p:.3f}")


if __name__ == "__main__":
    main()
