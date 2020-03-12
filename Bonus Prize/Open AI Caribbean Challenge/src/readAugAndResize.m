function im = readAugAndResize(filename,sz,mask_path)

im = imread(filename); 
im_s = strfind(filename, '\');
im_mask = imread(fullfile(mask_path, [strrep(filename(im_s(end)+1:end), '.png', '_mask.png')]));

if min([size(im_mask,1), size(im_mask,2)]) > 50
    [im, im_mask] = customAugmentationV2(im, im_mask);
end

im_r = im(:,:,1); 
im_g = im(:,:,2); 
im_b = im(:,:,3); 

im_r(im_mask == 0) = 0; 
im_g(im_mask == 0) = 0; 
im_b(im_mask == 0) = 0; 

im(:,:,1) = im_r; 
im(:,:,2) = im_g; 
im(:,:,3) = im_b; 

im = imresize(im, sz(1:2));

end