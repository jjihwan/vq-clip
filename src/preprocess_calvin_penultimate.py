from PIL import Image
import os
from glob import glob
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection
import decord
from tqdm import tqdm, trange
import argparse
import numpy as np


if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", "-r", type=int, default=0)
    parser.add_argument("--batch_size", "-b", type=int, default=15)

    args = parser.parse_args()
    bsz = args.batch_size

    cache_dir = "/131_data/jihwan/data/huggingface/hub/"

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir).to("cuda")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14", cache_dir=cache_dir)


    data_root = "/cvdata1/jihwan/calvin/dataset/"

    path_abcd = os.path.join(data_root, "ABCD")

    paths = [path_abcd]

    rank2char = {
        0: "A",
        1: "B",
        2: "C",
        3: "D"
    }
    for path in paths:
        files = sorted(glob(os.path.join(path, f"{rank2char[args.rank]}_*.mp4")))
        assert len(files) > 0
        print(f"Rank {args.rank}: Processing {len(files)} video")

        v_decoder = decord.VideoReader

        for file in files:
            output_path = file.replace("calvin/dataset", "calvin_lmdb").replace("/cvdata1/jihwan/", "/131_data/jihwan/data/").replace(".mp4", ".npy")
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)

            image_embedss = []

            vr = v_decoder(file)
            n_frames = len(vr)

            video_data = vr.get_batch(range(0, n_frames, 6)).asnumpy().transpose(0, 3, 1, 2)
            video_len = video_data.shape[0]
            print("video_len", video_len)

            inputs = processor(images=video_data, return_tensors='pt')
            input = inputs['pixel_values'].to('cuda')

            num_it = video_len // bsz if video_len % bsz == 0 else video_len // bsz + 1

            for i in trange(num_it):
                pixel_values = input[bsz*i:bsz*(i+1)]
                
                image_embeds = model(pixel_values=pixel_values, output_hidden_states=True).hidden_states[-2]
                # image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
                image_embeds = image_embeds.cpu().detach().numpy()

                image_embedss.append(image_embeds)
            
            image_embedss = np.concatenate(image_embedss, axis=0)

            print(f"Saving as {output_path}, {image_embedss.shape}")
            np.save(output_path, image_embedss)


                # for j in range(bsz):
                #     idx = i*bsz+j
                #     output_path = video_file.replace("minecraft", "minecraft_clip").replace(".mp4", f"_{idx:04d}.npy")
                #     output_path = output_path.replace("/cvdata1/jihwan/", "/131_data/jihwan/data/")
                #     np.save(output_path, image_embeds[j])



            
            