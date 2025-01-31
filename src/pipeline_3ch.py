import math
import torch
import numpy as np
import cv2
from torch import nn
from transformers import SamProcessor
from typing import Any

import matplotlib.pyplot as plt

# import the cellpose-style code:
from cellpose.dynamics import compute_masks
from slidingWindow import SlidingWindowHelper

class SlidingWindowPipeline3ch:
    def __init__(self, model, device, crop_size=256):
        self.model = model.get_model()
        self.device = device
        self.crop_size = crop_size
        self.processor = SamProcessor.from_pretrained('facebook/sam-vit-base')
        self.sliding_window_helper = SlidingWindowHelper(crop_size, 32)

    def _preprocess(self, img):
        if len(img.shape) != 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        inputs = self.processor(img, return_tensors="pt")
        return inputs['pixel_values'].to(self.device)

    def get_model_prediction(self, image):
        """
        Forward pass => returns 3 channels: (dx, dy, cell_prob).
        shape => [3, H, W] on CPU as a NumPy array.
        """
        image_input = self._preprocess(image)
        self.model.eval().to(self.device)

        with torch.no_grad():
            outputs = self.model(pixel_values=image_input, multimask_output=True)
            # shape => [B, 3, H, W], B=1 presumably
            pred_3ch = outputs.pred_masks.squeeze(1)  # e.g. shape [1, 3, h, w]
        pred_3ch = pred_3ch[0]  # remove batch dimension => shape [3, h, w]
        pred_3ch[2] = torch.sigmoid(pred_3ch[2]) # sigmoid on cell_prob

        # move to cpu => np
        pred_3ch_np = pred_3ch.detach().cpu().numpy()
        return pred_3ch_np

    def predict_on_full_img(self, image_orig):
        # 1) separate_into_crops
        crops, orig_regions, crop_unique_region = \
            self.sliding_window_helper.seperate_into_crops(image_orig)

        # 2) For each crop, run the model => shape [3, crop_size, crop_size]
        #    We'll store separate channel lists: one for dx, one for dy, one for prob
        dx_crops = []
        dy_crops = []
        prob_crops = []

        for crop in crops:
            pred_3ch = self.get_model_prediction(crop)  # shape => (3, crop_size, crop_size) on CPU
            # pred_3ch[0]: dx, pred_3ch[1]: dy, pred_3ch[2]: prob/distance
            dx_crops.append(pred_3ch[0])
            dy_crops.append(pred_3ch[1])
            prob_crops.append(pred_3ch[2])

        # 3) Now we do "combine_crops" for each channel individually
        #    But combine_crops expects shapes like (crop_size, crop_size),
        #    while dx_crops are also (crop_size, crop_size) => that’s correct.
        #    We just pass them as a list to combine.

        # Convert to float32 arrays of shape [B, Hc, Wc] first
        dx_crops = [dx.astype(np.float32) for dx in dx_crops]
        dy_crops = [dy.astype(np.float32) for dy in dy_crops]
        prob_crops = [p.astype(np.float32) for p in prob_crops]

        # call combine_crops => we do a slight trick: we feed them as "cropped_images"
        # each is shape => (crop_size, crop_size). Then we get a single (H, W) from each call.
        dx_full = self.sliding_window_helper.combine_crops(
            orig_size=image_orig.shape,  # shape => (H, W), or (H, W, 3)?
            cropped_images=dx_crops,  # the list of single-channel crops
            orig_regions=orig_regions,
            crop_unique_region=crop_unique_region
        )
        dy_full = self.sliding_window_helper.combine_crops(
            orig_size=image_orig.shape,
            cropped_images=dy_crops,
            orig_regions=orig_regions,
            crop_unique_region=crop_unique_region
        )
        prob_full = self.sliding_window_helper.combine_crops(
            orig_size=image_orig.shape,
            cropped_images=prob_crops,
            orig_regions=orig_regions,
            crop_unique_region=crop_unique_region
        )

        # If your image_orig is grayscale => shape (H, W), you want output => (H, W).
        # If image_orig is color => shape (H, W, 3). Then combine_crops returns (H, W, 3) for single channel?
        # => Possibly you'd have to handle that. In your code, it looks like you do
        #    "np.zeros(orig_size, dtype=np.float32)" => so if orig_size is (H, W), that’s good.
        #    If orig_size is (H, W, 3), you might want to pass just (H,W) as orig_size for the float maps.

        # => Now we have dx_full, dy_full, prob_full => each shape (H, W).
        # => We'll stack them => shape => (3, H, W)
        flows_3ch = np.stack([dx_full, dy_full, prob_full], axis=0)
        return flows_3ch

    def cells_from_flows(self, flows_3ch, cellprob_threshold):
        """
        flows_3ch => shape (3, H, W) => (dx, dy, cell_prob).
        We interpret channel 0 => dx, 1 => dy, 2 => cell_prob
        Then we call cellpose's compute_masks(dP, cellprob_threshold=whatever).
        """
        import torch
        import numpy as np
        from cellpose.dynamics import compute_masks

        dx = flows_3ch[0]
        dy = flows_3ch[1]
        cellprob = flows_3ch[2]

        # assemble dP with shape (2,H,W)
        dP = np.stack([dy, dx], axis=0)

        mask = compute_masks(
            dP=dP,
            cellprob=cellprob,
            niter=200,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=0,
            interp=True,
            do_3D=False,
            min_size=0,
            max_size_fraction=0.4,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        return mask

    def run(self, image, return_flows=False, sensitivity=5):
        """
        `sensitivity` in [1..10], default=5.
        We'll map it to a 'cellprob_threshold' in `compute_masks(...)`.
        We'll fill in some approximate logic for other values.
        """
        threshold_map = {
            1: 0.06,
            2: 0.07,
            3: 0.08,
            4: 0.09,
            5: 0.1,
            6: 0.11,
            7: 0.12,
            8: 0.13,
            9: 0.14,
            10: 0.15,
        }
        if sensitivity not in threshold_map:
            # fallback or clamp
            sensitivity = 5
        cellprob_thresh = threshold_map[sensitivity]

        # 1) get (dx,dy,prob)
        flows_3ch = self.predict_on_full_img(image)  # shape (3,H,W)

        # 2) do final instance labeling
        labels = self.cells_from_flows(
            flows_3ch=flows_3ch,
            cellprob_threshold=cellprob_thresh,
        )

        if return_flows:
            return labels, flows_3ch
        return labels

