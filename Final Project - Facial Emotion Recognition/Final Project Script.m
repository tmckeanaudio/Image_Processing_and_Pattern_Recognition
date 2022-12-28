%% Script for Figures, Equations, and Methods of IP&PR Final Project
% $$B_{pq} = \alpha_{p}\alpha_{q} \overset{M-1}\underset{m = 0}\sum \overset{N-1}\underset{n 
% = 0}\sum A_{mn} cos\left(\frac{(2m +1)u\pi}{2M}\right)cos\left(\frac{(2n +1)v\pi}{2N}\right)$$
% 
% with $0 \leq p \leq M-1$ and $0 \leq q \leq N-1$
% 
% and the constants $\alpha_u$ and $\alpha_v$ are
% 
% $$\alpha_u =  \left\{ \sqrt\frac{1}{M} \text{ for u = 0} \text{ and }   \sqrt\frac{2}{M} 
% \text{ for }1\leq u \leq M-1 $$
% 
% $$\alpha_v =  \left\{ \sqrt\frac{1}{N} \text{ for v = 0} \text{ and }   \sqrt\frac{2}{N} 
% \text{ for }1\leq v \leq M-1 $$
% 
% $$M \times n$$ $$M$$
% 
% $$A_{mn} $$ $$B_{pq}$$ $$p$$  $$q$$
% 
% $$2*n^2$$ 
% 
% $$D_j(x)=( (x-m_j) * (x-m_j)^T ) ^{1/2} \text{ }, \text{ } j = 1,2,3,... \text{ 
% },N_c$$
% 
% $$m_j$$ $$jth$$

f = intensityScaling(imread('HA5.tiff'));
n = 100;
[M,N] = size(f);
g = dct2(f);
figure
subplot(1,2,1), imshow(f), title("Face Image")
subplot(1,2,2), imshow(g), title("DCT Spectrum of Image")

mask = zeros(M,N);
mask(1:n,1:n) = (g(1:n,1:n));
h = idct2(mask);
figure,
subplot(1,2,1), imshow(f), title("Original Image")
subplot(1,2,2), imshow(h), title("Reconstructed using N = 100 Coefficients")
 
% Use Computer Vision Toolbox Functions to extract Face crop image
FaceDetect = vision.CascadeObjectDetector;
face_bboxes1 = step(FaceDetect, f);
% Crop face from original image 
face1 = imcrop(f,face_bboxes1);
g1 = dct2(face1);
figure
subplot(1,2,1),imshow(face1), title("Face Image")
subplot(1,2,2), imshow(g1), title("DCT Spectrum of Face")

EyeDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',7);
eyes_bboxes = step(EyeDetect,f);
% Crop Eyes from original image 
eyes = imcrop(f,eyes_bboxes);
e = dct2(eyes);

MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',150);
mouth_bboxes = step(MouthDetect,f);
% Crop Eyes from original image 
mouth = imcrop(f,mouth_bboxes);
m = dct2(mouth);

figure
subplot(2,2,1), imshow(eyes), title('Eye Region')
subplot(2,2,2), imshow(e), title('DCT of Eye Region')
subplot(2,2,3), imshow(mouth), title('Mouth Region')
subplot(2,2,4), imshow(m), title('DCT of Mouth Region')
%Detect Eyes and Mouth
EyeDetect = vision.CascadeObjectDetector('EyePairBig','MergeThreshold',7);
MouthDetect = vision.CascadeObjectDetector('Mouth','MergeThreshold',150);
eyes_bbox = step(EyeDetect,face);
mouth_bbox = step(EyeDetect,face);
% Annotate detected Eye Region
IFaces1 = insertObjectAnnotation(face, 'rectangle', eyes_bbox, 'Eyes');
IFaces2 = insertObjectAnnotation(face, 'rectangle', eyes_bbox, 'Mouth');
figure('Units','inches','Position',[0,0,12,4]);  
subplot(1,2,1), imshow(IFaces1)
subplot(1,2,2), imshow(IFaces2)

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