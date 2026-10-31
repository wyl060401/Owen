from typing import Dict, List
import PIL.Image

def load_pil_images(conversations: List[Dict[str, str]]) -> List[PIL.Image.Image]:
    """

    Args:
        conversations (List[Dict[str, str]]): the conversations with a list of messages. An example is :
            [
                {
                    "role": "User",
                    "content": "<image>
Extract all information from this image and convert them into markdown format.",
                    "images": ["./examples/table_datasets.png"]
                },
                {"role": "Assistant", "content": ""},
            ]

    Returns:
        pil_images (List[PIL.Image.Image]): the list of PIL images.

    """
    pil_images = []
    for message in conversations:
        if 'images' not in message:
            continue
        for image_path in message['images']:
            pil_img = PIL.Image.open(image_path)
            pil_img = pil_img.convert('RGB')
            pil_images.append(pil_img)
    return pil_images
