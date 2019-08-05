load('context.mat');
strTrainPath = 'C:\Users\priju75\Desktop\Image\image';
strTestPath = 'C:\Users\priju75\Desktop\Image\Test';


%NeutralImages=[];
%for i=1:length(imageLabel2{1,1})
%    if (strcmp(lower(imageLabel2{1,2}{i,1}),'neutral'))
%        NeutralImages=[NeutralImages,i];
%    end 
%end
%if (length(NeutralImages)==0)
%    disp('ERROR: Neutral Expression is not available in training');
%    return;
%end

structTestImages = dir(strTestPath);
numImage = length(imageLabel2{1,1});  % Total Observations: Number of Images in training set
lenTest = length(structTestImages);

if (lenTest==0)
    disp('Error:Invalid Test Folder');
    return;
end

TrainImages='';
for i = 1:numImage
	TrainImages{i,1} = strcat(strTrainPath,'\',imageLabel2{1,1}(i));
end

j=0;
for i = 3:lenTest
     if ((~structTestImages(i).isdir))
         if  (structTestImages(i).name(end-3:end)=='.jpg')
             j=j+1;
             TestImages{j,1} = [strTestPath,'\',structTestImages(i).name];
         end
     end
end
numTestImage = j; % Number of Test Images
clear ('structTestImages','fid','i','j');pack

imageSize = [280,180];          % All Images are resized into a common size



img = zeros(imageSize(1)*imageSize(2),numImage);
for i = 1:numImage
    img1=imread(cell2mat(TrainImages{i,1}));
    detector=vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
    BB=step(detector,img1);
    face=imcrop(img1,BB(1,:));
    face=imresize(face,imageSize);
    if size(face,3)==3
        face=rgb2gray(face);
    end
    img(:,i) = face(:);
    disp(sprintf('Loading Train Image # %d',i));
end
meanImage = mean(img,2);        
                 
img = (img - meanImage*ones(1,numImage))';      % img is the input to PCA



[C,S,L]=princomp(img,'econ');                   % Performing PCA Here
EigenRange = [1:30];   % Defines which Eigenvalues will be selected
C = C(:,EigenRange);




img = zeros(imageSize(1)*imageSize(2),numTestImage);
for i = 1:numTestImage
    img1=imread(cell2mat(TrainImages{i,1}));
    detector=vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
    BB=step(detector,img1);
    face=imcrop(img1,BB(1,:));
    face=imresize(face,imageSize);
    if size(face,3)==3
        face=rgb2gray(face);
    end
    img(:,i) = face(:);
    disp(sprintf('Loading Test Image # %d',i));
end
%meanImage = mean(img,2);        
img = (img - meanImage*ones(1,numTestImage))';
Projected_Test = img*C;


EucDist = zeros(numTestImage,numImage);
for projectedImgIndex = 1:numTestImage
    TestImage = Projected_Test(projectedImgIndex,:);
    for i = 1:numImage
        EucDist(projectedImgIndex,i) = sqrt((TestImage'-S(i,EigenRange)')' ...
            *(TestImage'-S(i,EigenRange)'));
    end
end
[Min_Dist,Min_Dist_pos] = min(EucDist,[],2);


%meanImage1=(meanImage'*C)';
%%meanNutral = mean(S(NeutralImages,EigenRange)',2);
%for Dat2Project = 1:numTestImage
%    TestImage = Projected_Test(Dat2Project,:);
%    % Picking the image #Dat2Project
% 
%    Eucl_Dist(Dat2Project) = sqrt((TestImage'-meanImage1)'*(TestImage' ...
%        -meanImage1));
%        % Here, the distance between the expression under test and
%        % the mean neutral expressions is being calculated
%end
%%Eucl_Dist = Eucl_Dist/max(Eucl_Dist);



%Other_Dist = zeros(numTestImage,numImage);
%for Dat2Project = 1:numTestImage
%   TestImage = Projected_Test(Dat2Project,:);
%    % Picking the image #Dat2Project
%    for i = 1:numImage
%        Other_Dist(Dat2Project,i) = sqrt((TestImage'-S(i,EigenRange)')' ...
%            *(TestImage'-S(i,EigenRange)'));
%    end
%end
%[Min_Dist,Min_Dist_pos] = min(Other_Dist,[],2);



fid = fopen('Results.txt','w');
fprintf(fid,'//Test Image,Distance From Neutral, Expression,Best Match\r\n');

for i = 1:numTestImage
    b = find(TestImages{i,1}=='\');
    Test_Image = TestImages{i,1}(b(end)+1:end);
%    Dist_frm_Neutral = Eucl_Dist(i);
    Best_Match = cell2mat(imageLabel2{1,1}(Min_Dist_pos(i)));
    Expr = cell2mat(imageLabel2{1,2}(Min_Dist_pos(i)));
    fprintf(fid,'%s,%s,%s\r\n',Test_Image,Expr,Best_Match);
end
fclose(fid);

isSucceed = 1;
disp('Done')
disp('Output File = .\Results.txt');