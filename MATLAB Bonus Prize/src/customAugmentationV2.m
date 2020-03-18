function [image_augmented, mask_augmented] = customAugmentationV2(image, image_mask)
%% Rotation
tform = randomAffine2d('Rotation',[-180 180]); 
outputView = affineOutputView(size(image),tform,'BoundsStyle','FollowOutput');
image_augmented = imwarp(image,tform,'OutputView',outputView);  

outputView_mask = affineOutputView(size(image_mask),tform,'BoundsStyle','FollowOutput');
mask_augmented = imwarp(image_mask,tform,'OutputView',outputView_mask);  
 

%% Reflection
tform = randomAffine2d('XReflection',true,'YReflection',true);
outputView = affineOutputView(size(image_augmented),tform);
image_augmented = imwarp(image_augmented,tform,'OutputView',outputView);

outputView_mask = affineOutputView(size(mask_augmented),tform,'BoundsStyle','FollowOutput');
mask_augmented = imwarp(mask_augmented,tform,'OutputView',outputView_mask);  

%% Shear
tform = randomAffine2d('XShear',[-5 5]); 
outputView = affineOutputView(size(image_augmented),tform,'BoundsStyle','FollowOutput');
image_augmented = imwarp(image_augmented,tform,'OutputView',outputView);


outputView_mask = affineOutputView(size(mask_augmented),tform,'BoundsStyle','FollowOutput');
mask_augmented = imwarp(mask_augmented,tform,'OutputView',outputView_mask);  
 

%% Final result

[row_list, col_list] = find(mask_augmented > 0); 

image_augmented = image_augmented(min(row_list):max(row_list), min(col_list):max(col_list), :); 
mask_augmented = mask_augmented(min(row_list):max(row_list), min(col_list):max(col_list)); 