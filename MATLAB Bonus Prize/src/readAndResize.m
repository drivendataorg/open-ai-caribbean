function im = readAndResize(filename,sz,mask_path)

%% Using the mask to remove unrelated pixels
im = imread(filename); 
im_s = strfind(filename, '\');
im_mask = imread(fullfile(mask_path, [strrep(filename(im_s(end)+1:end), '.png', '_mask.png')]));

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