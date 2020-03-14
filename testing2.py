import pandas as pd
import numpy as np
import ImageTransformer
import ImageTransformer as imgt
import string

data = np.random.random((10, 11))

for idx in range(10):
    data[idx][10] = idx % 2
# print(data)

train_x = pd.DataFrame(data[:, :10], columns=[x for x in string.ascii_lowercase[:10]])
train_y = pd.DataFrame(data[:, 10])
print(train_x, train_y, sep='\n')

iat22 = train_x.iat[2,2]
iat82 = train_x.iat[8,2]
# at82 = train_x.at[8,2]



# applying and plotting k-pca. for each feature in the training, assign X and Y as coordinates
kpca_point = imgt.dimension_reduction(kernel='cosine', features_df=train_x)
imgt.plot_scatter(kpca_point)

# constructing the image
features_to_pixels = imgt.divide_to_pixels(scatter=kpca_point, resolution=10)
pixels_to_features = imgt.to_pixel_map(features_to_pixels, resolution=10)
pixels_features_heat_map = imgt.feat_count_per_pixel(resolution=10, feat_in_pixels=features_to_pixels)
imgt.plot_heat_map(pixels_features_heat_map, fig_size=10)

print()