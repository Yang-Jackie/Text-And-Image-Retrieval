import faiss
import matplotlib.pyplot as plt
import math
import numpy as np
import open_clip
from langdetect import detect


class Myfaiss:
    def __init__(
        self,
        bin_file: str,
        id2img_fps,
        device,
        translater,
        clip_backbone="ViT-SO400M-14-SigLIP-384",
        clip_pretrained="webli",
    ):
        self.index = self.load_bin_file(bin_file)
        self.id2img_fps = id2img_fps
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_backbone, pretrained=clip_pretrained, device=device
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-SO400M-14-SigLIP-384')
        self.translater = translater

    def load_bin_file(self, bin_file: str):
        return faiss.read_index(bin_file)

    def show_images(self, image_paths):
        fig = plt.figure(figsize=(15, 10))
        columns = int(math.sqrt(len(image_paths)))
        rows = int(np.ceil(len(image_paths) / columns))

        for i in range(1, columns * rows + 1):
            img = plt.imread(image_paths[i - 1])
            ax = fig.add_subplot(rows, columns, i)
            ax.set_title("/".join(image_paths[i - 1].split("/")[-3:]))

            plt.imshow(img)
            plt.axis("off")

        plt.show()

    def image_search(self, id_query, k):
        query_feats = self.index.reconstruct(id_query).reshape(1, -1)

        scores, idx_image = self.index.search(query_feats, k=k)
        idx_image = idx_image.flatten()

        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]

        return scores, idx_image, infos_query, image_paths

    def text_search(self, text, k):
        if detect(text) == "vi":
            text = self.translater(text)

        ###### TEXT FEATURES EXACTING ######
        text = self.tokenizer([text]).to(self.device)
        text_features = (
            self.model.encode_text(text).cpu().detach().numpy().astype(np.float32)
        )

        ###### SEARCHING #####
        scores, idx_image = self.index.search(text_features, k=k)
        idx_image = idx_image.flatten()

        ###### GET INFOS KEYFRAMES_ID ######
        infos_query = list(map(self.id2img_fps.get, list(idx_image)))
        image_paths = [info for info in infos_query]

        return scores, idx_image, infos_query, image_paths
