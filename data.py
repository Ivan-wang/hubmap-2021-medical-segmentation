import pandas as pd
import zarr
import cv2


ght_dict = {
    'fbr': 0.01,          # Background Weights
    'cortex_value': 0.5,  # Cortex Weights
    'medulla_value': 0.25 # Medulla Weights
}
def convert_region_to_pdf(resize=512):
    zarr_mask = zarr.open('./data/zarr_mask', mode='a')
    zarr_pdf = zarr_mask.create_group('pdfs', overwrite=True)

    zarr_label = zarr_mask['label']
    zarr_region = zarr_mask['region']

    traindf = pd.read_csv(TRAINDF)
    for i in tqdm(range(traindf.shape[0])):
        iid = traindf.iloc[i, 0]
        r = zarr_region[iid][:]
        mask = zarr_region[iid][:]
        prob = np.zeros_like(r, dtype=np.float32)
        prob[r == 0] = weight_dict['fbr']
        prob[r == 1] = weight_dict['cortex_value']
        prob[r > 1] = weight_dict['medulla_value']
        prob[mask > 0] = 1.

        del r
        del mask
        gc.collect()

        # NOTE: the same resize parameter should be used in RandomTileDataset
        if prob.shape[0]> resize:
            resize_w = int((prob.shape[1]/prob.shape[0])*resize)
            prob = cv2.resize(prob.astype('float32'), dsize=(resize_w, resize))
        prob = prob / prob.sum()
        prob = np.cumsum(prob, dtype=np.float64)
        prob = prob.astype(np.float32)
        zarr_pdf[iid] = prob

        del prob
        gc.collect()

def check_pdf():
    zarr_pdf = zarr.open('./data/zarr_mask/pdfs', mode='r')
    zarr_r = zarr.open('./data/zarr_mask/region', mode='r')
    zarr_mask = zarr.open('./data/zarr_mask/label', mode='r')
    k = list(zarr_pdf.keys())[0]
    print('Image ID:', k)
    print('Region Shape:', zarr_r[k].shape)
    print('Mask Shape:', zarr_mask[k].shape)
    print('PDF Shape:', zarr_pdf[k].shape)

if __name__ == '__main__':
    check_pdf()
