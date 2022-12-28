%% Live Script for Testing Facial Expression Recognition Algorithm
% Process for Facial Expression Recognition is as follows:
%% 
% * Input Labeled Images into workspace and categorize them based on 5 emotions: 
% Happy, Sad, Angry, Surprised, Fear, and Neutral
% * Make images Monochrome if they are RGB images, and intensity scale the images 
% to have pixel values between 0 and 1
% * Preprocess the images by resizing them and then cropping them to be a determined 
% M x N sized image
% * Extract the two main areas of interest of the faces, which are the Eye/Eyebrow 
% and Mouth regions
% * Once regions are extracted from original image, use the 2D Discrete Cosine 
% Transform to extract low frequency coefficients that happen to capture the most 
% relevant info about the facial expressions
% * 15 LF coefficients from both the Eye/Mouth region of one image will be combined 
% into a single array and used to be input into the Classification Learner App 
% inside MATLAB
% * Once the Neural Network is trained using enough images, output the trained 
% model and instantiate it as a module within the custom GUI  for our project
% * Ideally, the GUI will allow the user to upload a new raw image and will 
% follow the exact steps laid out above and result in a classification output 
% image of one of the 5/6 emotions will some degree of confidence.
% * The CI we are hoping to achieve should be around 95%, but if we run out 
% of time, we will present methods of optimizing the model to a obtain a higher 
% degree of CI
%% Test Preprocessing Images by Cropping Faces

clc; close all; clear;
f1 = intensityScaling(imread("girl.tif"));
f2 = intensityScaling(rgb2gray4e(imread('ballerina.tif')));
figure('Units','inches','Position',[0,0,12,4]); 
subplot(1,2,1), imshow(f1)
subplot(1,2,2), imshow(f2)
faceDetector = vision.CascadeObjectDetector; % Default: finds faces
bboxes1 = step(faceDetector, f1); % Detect faces
bboxes2 = step(faceDetector, f2); % Detect faces
% Annotate detected faces
IFaces1 = insertObjectAnnotation(f1, 'rectangle', bboxes1, 'Face');
IFaces2 = insertObjectAnnotation(f2, 'rectangle', bboxes2, 'Face');
figure('Units','inches','Position',[0,0,12,4]); 
subplot(1,2,1), imshow(IFaces1), title('Detected Face Girl Image');
subplot(1,2,2), imshow(IFaces2), title('Detected Face Ballerina Image');
croppedf1 = imcrop(f1,bboxes1);
croppedf2 = imcrop(f2,bboxes2);
figure('Units','inches','Position',[0,0,12,4]); 
subplot(1,2,1), imshow(croppedf1), title('Cropped Girl Face');
subplot(1,2,2), imshow(croppedf2), title('Cropped Ballerina Face');
%% Test Preprocessing Images by Cropping Eye and Mouth Regions

clc; close all; clear;
girl = imread("girl.tif");
ballerina = rgb2gray4e(imread('ballerina.tif'));
figure('Units','inches','Position',[0,0,12,4]); 
subplot(1,2,1), imshow(girl), title('Girl Image');
subplot(1,2,2), imshow(ballerina), title('Ballerina Image');
%Detect Eyes
EyeDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',7);
eyes_bboxes1 = step(EyeDetect,girl);
eyes_bboxes2 = step(EyeDetect,ballerina);
% Annotate detected Eye Region
IFaces1 = insertObjectAnnotation(girl, 'rectangle', eyes_bboxes1, 'Eyes');
IFaces2 = insertObjectAnnotation(ballerina, 'rectangle', eyes_bboxes2, 'Eyes');
figure('Units','inches','Position',[0,0,12,4]);  
subplot(1,2,1), imshow(IFaces1), title('Girl Image');
subplot(1,2,2), imshow(IFaces2), title('Ballerina Image');
% Crop Eye Regions
eyes_girl = imcrop(girl,eyes_bboxes1);
eyes_ballerina = imcrop(ballerina,eyes_bboxes2);

figure('Units','inches','Position',[0,0,12,4]); 
subplot(1,2,1), imshow(eyes_girl), title('Eye Region of Girl Image');
subplot(1,2,2), imshow(eyes_ballerina), title('Eye Region of Ballerina Image');
% Detect Mouth
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',150);
mouth_bboxes1 = step(MouthDetect,girl);
mouth_bboxes2 = step(MouthDetect,ballerina);
mouth_bboxes1(1) = mouth_bboxes1(1) + 5;
mouth_bboxes2(1) = mouth_bboxes2(1) - 15;
mouth_bboxes2(3) = mouth_bboxes2(3) + 15;
% Annotate detected Mouth Region
IFaces1 = insertObjectAnnotation(girl, 'rectangle', mouth_bboxes1, 'Mouth');
IFaces2 = insertObjectAnnotation(ballerina, 'rectangle', mouth_bboxes2, 'Mouth');
figure('Units','inches','Position',[0,0,12,4]);  
subplot(1,2,1), imshow(IFaces1), title('Girl Image');
subplot(1,2,2), imshow(IFaces2), title('Ballerina Image');
% Crop Mouth Region
mouth_girl = imcrop(girl,mouth_bboxes1);
mouth_ballerina = imcrop(ballerina,mouth_bboxes2);
figure('Units','inches','Position',[0,0,12,4]);  
subplot(1,2,1), imshow(mouth_girl), title('Detected Mouth Region of Girl Image');
subplot(1,2,2), imshow(mouth_ballerina), title('Detected Mouth Region of Ballerina Image');
%% Perform 2D Discrete Cosine Transform to extract DCT coefficients

DCT_mouth_girl = dct2(mouth_girl);
DCT_eyes_girl = dct2(eyes_girl);
DCT_mouth_ballerina = dct2(mouth_ballerina);
DCT_eyes_ballerina = dct2(eyes_ballerina);
figure('Units','inches','Position',[0,0,12,4]);  
subplot(2,2,1); imshow(DCT_mouth_girl)
subplot(2,2,2); imshow(DCT_eyes_girl)
subplot(2,2,3); imshow(DCT_mouth_ballerina)
subplot(2,2,4); imshow(DCT_eyes_ballerina)
% Try Extracting a 4 x 4 grid of the lowest DCT coefficients, so 16 low frequency values, from both Eye and Mouth Regions for both images

Mouthtable = zeros(2,16);
Mouthtable(1,1:4) = DCT_mouth_girl(1,1:4);
Mouthtable(1,5:8) = DCT_mouth_girl(2,1:4);
Mouthtable(1,9:12) = DCT_mouth_girl(3,1:4);
Mouthtable(1,13:16) = DCT_mouth_girl(4,1:4);
Mouthtable(2,1:4) = DCT_mouth_ballerina(1,1:4);
Mouthtable(2,5:8) = DCT_mouth_ballerina(2,1:4);
Mouthtable(2,9:12) = DCT_mouth_ballerina(3,1:4);
Mouthtable(2,13:16) = DCT_mouth_ballerina(4,1:4);
Mouthtable
Eyetable = zeros(2,16);
Eyetable(1,1:4) = DCT_eyes_girl(1,1:4);
Eyetable(1,5:8) = DCT_eyes_girl(2,1:4);
Eyetable(1,9:12) = DCT_eyes_girl(3,1:4);
Eyetable(1,13:16) = DCT_eyes_girl(4,1:4);
Eyetable(2,1:4) = DCT_eyes_ballerina(1,1:4);
Eyetable(2,5:8) = DCT_eyes_ballerina(2,1:4);
Eyetable(2,9:12) = DCT_eyes_ballerina(3,1:4);
Eyetable(2,13:16) = DCT_eyes_ballerina(4,1:4);
Eyetable
DCT_table = horzcat(Eyetable,Mouthtable);
DCT_table = array2table(DCT_table);
DCT_table.Properties.VariableNames = {'Eye C1' 'Eye C2' 'Eye C3' 'Eye C4' 'Eye C5' 'Eye C6' 'Eye C7' 'Eye C8' 'Eye C9' 'Eye C10' 'Eye C11' 'Eye C12' 'Eye C13' 'Eye C14' 'Eye C15' 'Eye C16' 'Mouth C1' 'Mouth C2' 'Mouth C3' 'Mouth C4' 'Mouth C5' 'Mouth C6' 'Mouth C7' 'Mouth C8' 'Mouth C9' 'Mouth C10' 'Mouth C11' 'Mouth C12' 'Mouth C13' 'Mouth C14' 'Mouth C15' 'Mouth C16'};
DCT_table.EmotionClass = cell(2,1);
DCT_table.EmotionClass(1) = {'Girl'};
DCT_table.EmotionClass(2) = {'Ballerina'};
DCT_table.EmotionClass = categorical(DCT_table.EmotionClass);
DCT_table
%% Extract DCT Coefficients from 5 basic emotion images and input into Classification Learner
% Start first by extracting the face DCT coefficients

n = 5;
a = imread('happy.png');
b = rgb2gray4e(imread('sadness.png')); 
c = imread('anger.png'); 
d = imread('disgust.png'); 
e = rgb2gray4e(imread('fear.png'));

featureFaceTable = zeros(5,n^2);
featureFaceTable(1,1:n^2) = extractDCT_Face(a,n);
featureFaceTable(2,1:n^2) = extractDCT_Face(b,n);
featureFaceTable(3,1:n^2) = extractDCT_Face(c,n);
featureFaceTable(4,1:n^2) = extractDCT_Face(d,n);
featureFaceTable(5,1:n^2) = extractDCT_Face(e,n);
featureFaceTable = array2table(featureFaceTable);
for i = 1:n^2
    fname = ['B',num2str(i)];
    featureFaceTable.Properties.VariableNames(i) = {fname};
end

featureFaceTable.EmotionClass = cell(5,1);
featureFaceTable.EmotionClass = {'happy'; 'sad'; 'anger'; 'disgust'; 'fear'}
featureFaceTable.EmotionClass = categorical(featureFaceTable.EmotionClass);
%%
a = [1,2,4; 3,5,7; 6,8,9;];
b = zigzag(a)
%% Create a Difference Image for DCT coefficient extraction

n = 8;
f = imread('HA2.tiff');
g = imread('NE2.tiff');
% Use Computer Vision Toolbox Functions to extract Face crop image
FaceDetect = vision.CascadeObjectDetector;
face_bboxes1 = step(FaceDetect, f);
face_bboxes2 = step(FaceDetect, g);
% Crop face from original image 
face1 = imcrop(f,face_bboxes1);
face2 = imcrop(g,face_bboxes2);
face1 = imresize(face1,[128,128]);
face2 = imresize(face2,[128,128]);
difface = intensityScaling(imsubtract(face1,face2));
figure('Units','inches','Position',[0,0,12,4]); 
subplot(1,3,1),imshow(face1);
subplot(1,3,2),imshow(face2);
subplot(1,3,3),imshow(difface);
DCT_face = dct2(difface);
% Extract an n x n block from the low frequencies of the DCT coefficients
facetable = DCT_face(1:n,1:n);
% Once the DCT Matrix is extracted, calculate the mean and standard
% deviation for the row-column-diagonal of the matrix and result in a
% feature vector for input into a KNN model
featureVector = zigzag(facetable);
% Generate plot of Image with DCT-II transform image
figure('Units','inches','Position',[0,0,12,4])
subplot(1,2,1),imshow(DCT_face);
subplot(1,2,2), plot(0:length(featureVector)-1,featureVector);