clear all;
load('context.mat');
strTrainPath='I:\Image\image';
TrainImages='';
imageLabel1='';
for i = 1:162
	TrainImages{i,1} = strcat(strTrainPath,'\',imageLabel2{1,1}(i));
end
a=[];
for i = 1:162
    xyz=TrainImages{i,1};
	img=imread(xyz{1});
    detector=vision.CascadeObjectDetector('ClassificationModel','FrontalFaceLBP');
    BB=step(detector,img);
    face = (img(BB(1,2):BB(1,2)+BB(1,4),BB(1,1):BB(1,1)+BB(1,3),:));
    %face=imcrop(img,BB(1,:));
    face=imresize(face,[125,100]);
    if size(face,3)==3
        face=rgb2gray(face);
    end
    feature=extractLBPFeatures(face,'CellSize',[32 32],'Radius',5);
    a=[a;feature];
end
trainmat(1:10,1:531)=0;
happy=0;
sad=0;
angry=0;
disgust=0;
surprised=0;
very_happy=0;
very_sad=0;
very_angry=0;
very_disgust=0;
very_surprised=0;
for i=1:162
    abc=imageLabel2{1,2}(i);
    if strcmp(abc,'happy')
        trainmat(1,:)=trainmat(1,:)+a(i,:);
        happy=happy+1;
    end
    if strcmp(abc,'disgust')
        trainmat(2,:)=trainmat(2,:)+a(i,:);
        disgust=disgust+1;
    end
    if strcmp(abc,'angry')
        trainmat(3,:)=trainmat(3,:)+a(i,:);
        angry=angry+1;
    end
    if strcmp(abc,'sad')
        trainmat(4,:)=trainmat(4,:)+a(i,:);
        sad=sad+1;
    end
    if strcmp(abc,'surprised')
        trainmat(5,:)=trainmat(5,:)+a(i,:);
        surprised=surprised+1;
    end
    if strcmp(abc,'very_happy')
        trainmat(6,:)=trainmat(6,:)+a(i,:);
        very_happy=very_happy+1;
    end
    if strcmp(abc,'very_disgust')
        trainmat(7,:)=trainmat(7,:)+a(i,:);
        very_disgust=very_disgust+1;
    end
    if strcmp(abc,'very_angry')
        trainmat(8,:)=trainmat(8,:)+a(i,:);
        very_angry=very_angry+1;
    end
    if strcmp(abc,'very_sad')
        trainmat(9,:)=trainmat(9,:)+a(i,:);
        very_sad=very_sad+1;
    end
    if strcmp(abc,'very_surprised')
        trainmat(10,:)=trainmat(10,:)+a(i,:);
        very_surprised=very_surprised+1;
    end
end
trainmat(1,:)=trainmat(1,:)/happy;
trainmat(2,:)=trainmat(2,:)/disgust;
trainmat(3,:)=trainmat(3,:)/angry;
trainmat(4,:)=trainmat(4,:)/sad;
trainmat(5,:)=trainmat(5,:)/surprised;
trainmat(6,:)=trainmat(6,:)/very_happy;
trainmat(7,:)=trainmat(7,:)/very_disgust;
trainmat(8,:)=trainmat(8,:)/very_angry;
trainmat(9,:)=trainmat(9,:)/very_sad;
trainmat(10,:)=trainmat(10,:)/very_surprised;

test=imread('Image11.jpg');
BB=step(detector,test);
face = (test(BB(1,2):BB(1,2)+BB(1,4),BB(1,1):BB(1,1)+BB(1,3),:));
%face=imcrop(test,BB(1,:));
face=imresize(face,[125,100]);
if size(face,3)==3
    face=rgb2gray(face);
end
feature=extractLBPFeatures(face,'CellSize',[32 32],'Radius',5);

distance(1:10)=[0 0 0 0 0 0 0 0 0 0];
distance(1)=norm(trainmat(1,:)-feature);
distance(2)=norm(trainmat(2,:)-feature);
distance(3)=norm(trainmat(3,:)-feature);
distance(4)=norm(trainmat(4,:)-feature);
distance(5)=norm(trainmat(5,:)-feature);
distance(6)=norm(trainmat(6,:)-feature);
distance(7)=norm(trainmat(7,:)-feature);
distance(8)=norm(trainmat(8,:)-feature);
distance(9)=norm(trainmat(9,:)-feature);
distance(10)=norm(trainmat(10,:)-feature);
disp(Image07)