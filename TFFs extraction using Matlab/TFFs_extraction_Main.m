%-------------------------------File info---------------------------------%
% - File name: TFFs_extraction_Main.m
% - Description:  
% - Input:  None
% - Output:  TFFs
% - Calls: ........
% - usage:  Just run the file.
% - Version： V1.0
% - Last update: 2020-09-26
% - Copyright (C) ASVP, South China university of technology; 2020
%  ------For Educational and Academic Purposes Only ------ 
% - Author : Chester.Wei.Xie, PRMI, SCUT/ GXU
% - Contact: chester.w.xie@gmail.com
%-------------------------------------------------------------------------%


%------------------- Clean up the environment first -----------------------
clear;
close all; 
clc;
%-------------------------- Include folders  -----------------------------%
addpath(genpath('Data'))                   %- Data folder 
                         
addpath(genpath('Functions'))           %-  Basic function

addpath(genpath('Application'))         %- Application (Base on the functions)

addpath(genpath('Test'))                   %- Testing....

addpath(genpath('Results'))              %- Save the results

%--------------------------Print startup ----------------------------------
fprintf(['\n',...
    'Welcome to:  ASVP Lab.\n', ...
    'Project: DMA-Net for ASC, the dataset used in this project including:  ..\n', ...
    'DCASE 2016/2017/2018-1A/20191A/2021-1A and LITIS_Rouen \n\n',...
    'You may contact Dr. Chester.W.Xie if you have any questions. \n\n']);
fprintf('                   -------------------------------\n\n');


feature_base_dir = fullfile('Results');

TF_method={'STFT','CQT'};
% TF_method={'STFT','CQT','CWT','Scattering'};

%% -Data Preparation

[train_set,test_set]=get_meta_dcase_2018_1a();


for index=1:numel(TF_method)
    TFRs_extration_for_dataset(train_set,TF_method{index},feature_base_dir);
    TFRs_extration_for_dataset(test_set, TF_method{index},feature_base_dir);
end






%% 
function [train_set,evaluate_set]=get_meta_dcase_2018_1a()
    folder = '.\Data';     %- Set folder to the location of the downloaded dataset.

    %- Read in the development set metadata as a table. Name the table variables FileName, AcousticScene.
    metadata_train = readtable([folder,'/TUT-urban-acoustic-scenes-2018-development/evaluation_setup/fold1_train.txt'], ...
        'Delimiter',{'\t'}, ...
        'ReadVariableNames',false);
    metadata_train.Properties.VariableNames = {'FileName','AcousticScene'};
    head(metadata_train)

    metadata_evaluate = readtable([folder,'/TUT-urban-acoustic-scenes-2018-development/evaluation_setup/fold1_evaluate.txt'], ...
        'Delimiter',{'\t'}, ...
        'ReadVariableNames',false);
    metadata_evaluate.Properties.VariableNames = {'FileName','AcousticScene'};
    head(metadata_evaluate)


    %- The first variable of the metadata tables contains the file names. Concatenate the file names with the file paths.
    train_datafolder = [folder,'/TUT-urban-acoustic-scenes-2018-development'];
    train_filePaths = strcat(train_datafolder,'/',metadata_train.FileName);

    evaluate_datafolder = [folder,'/TUT-urban-acoustic-scenes-2018-development'];
    evaluate_filePaths = strcat(evaluate_datafolder,'/',metadata_evaluate.FileName);

    %-------------------------------------------------------------------------%
    %- Create audio datastores for the train and test sets. Set the Labels property 
    %- of the audioDatastore to the acoustic scene. Call countEachLabel to verify an 
    %- even distribution of labels in both the train and test sets.
    %-------------------------------------------------------------------------%
    train_set = audioDatastore(train_filePaths, ...
        'Labels',categorical(metadata_train.AcousticScene));
    display(countEachLabel(train_set))

    evaluate_set = audioDatastore(evaluate_filePaths, ...
        'Labels',categorical(metadata_evaluate.AcousticScene));
    display(countEachLabel(evaluate_set))

end


%% 
% -
function [] =TFRs_extration_for_dataset(ads,Transform_method,feature_base_dir)
    
    [num_files,~]=size(ads.Files);
    reset(ads)
    Audio_path_list=ads.Files;
    % Creation of a timer waitbar object
    ppm = ParforProgressbar(num_files,'showWorkerProgress', true);

    parfor i=1:num_files
        [audio,fs] = audioread(Audio_path_list{i});
        info_FileName=Audio_path_list{i};

        TFRs_extration_for_audio(audio,fs,info_FileName,Transform_method,feature_base_dir);       
         % Timer waitbar update
        ppm.increment();

    end
    % Object deletion
    delete(ppm); 

    fprintf('Done.\n')
end

    
%%
function [Results_save_path] =TFRs_extration_for_audio(audio,fs,info_FileName,Transform_method,feature_base_dir)
%% 
 %-  
    dataMidSide= [sum(audio,2),audio(:,1)-audio(:,2)];
    dataMidSide=dataMidSide./max((max(dataMidSide)));

    originalFs = fs;
    desiredFs = 22050;

    [p,q] = rat(desiredFs / originalFs);
    inData = resample(dataMidSide,p,q); % - 
    fs=desiredFs;

    % - 
    
    filefullname=info_FileName; % - 
    [file_path,file_name,~]=fileparts(filefullname);  %- 
    Results_Dir=strrep(file_path,'Data',feature_base_dir);
    Results_Dir=strrep(Results_Dir,'audio',Transform_method);
    % -  

    Results_save_path=Results_Dir;  %- 
    if ~exist(Results_save_path,'dir')==1
         mkdir(Results_save_path);         
    end
    
    switch Transform_method
        case 'STFT'
              STFT_based_TFRs_extration(inData,fs,file_name,Results_save_path);  
        case 'CQT'
              CQT_based_TFRs_extration(inData,fs,file_name,Results_save_path);  
        case 'CWT'
              CWT_based_TFRs_extration(inData,fs,file_name,Results_save_path);
        case 'Scattering'
            Scattering_extration(inData,fs,file_name,Results_save_path);
    end


end



function []=STFT_based_TFRs_extration(inData,fs,file_name,Results_save_path)
 %%    
    % - 
    Results_dir_Mel=fullfile(Results_save_path,'Mel');
    Results_dir_Bark=fullfile(Results_save_path,'Bark');
    Results_dir_Gammatone=fullfile(Results_save_path,'Gammatone');

    if ~exist(Results_dir_Mel,'dir')==1
         mkdir(Results_dir_Mel);         
    end
    if ~exist(Results_dir_Bark,'dir')==1
         mkdir(Results_dir_Bark);         
    end
    if ~exist(Results_dir_Gammatone,'dir')==1
         mkdir(Results_dir_Gammatone);         
    end
    % - 
    win = hann(2048,"periodic"); 
    noverlap = 1024;
    fftLength = 1024*2;

    [fb_mel,~] = designAuditoryFilterBank(fs, ...
                "FrequencyScale","mel", ...
                "FFTLength",fftLength,...
                "NumBands",256);   

    [fb_bark,~] = designAuditoryFilterBank(fs, ...
                "FrequencyScale","bark", ...
                "FFTLength",fftLength,...
                "NumBands",256);

  
   gammaFiltBank = gammatoneFilterBank('SampleRate',fs, ...
                                        'NumFilters',256);% - audible range 
    

    [S,~,~] = stft(inData,fs, ...
           "Window",win, ...
           "OverlapLength",noverlap, ...
           "FFTLength",fftLength, ...
           "FrequencyRange","onesided");
    PowerSpectrum = S.*conj(S);  
    
    % To apply frequency-domain filtering, perform a matrix multiplication of the filter bank and the power spectrogram. 

    for i=1:2
        temp_mel=fb_mel * PowerSpectrum(:,:,i);
        temp_bark=fb_bark * PowerSpectrum(:,:,i);

        temp_mel=20*log10(temp_mel+eps);
        temp_bark=20*log10(temp_bark+eps); % - 
        
        temp_mel(:,end)=[];
        temp_bark(:,end)=[];
        

        spec_mel_dB(:,:,i)=temp_mel;   % -  
        spec_bark_dB(:,:,i)=temp_bark;
        
        audioOut = gammaFiltBank(inData(:,i));  
        buff = dsp.AsyncBuffer(numel(inData(:,i)));
        write(buff,audioOut.^2);
        sink = dsp.AsyncBuffer(numel(inData(:,i)));
        while buff.NumUnreadSamples > 0
            currentFrame = read(buff,2048,1024);
            write(sink,mean(currentFrame,1));
        end  

        % Convert the energy values to dB. Plot the energy-per-band over time.
        gammatoneSpec = read(sink);
        temp_spec = 20*log10(gammatoneSpec'+eps);  %-  

        spec_gammatone_dB(:,:,i)=temp_spec;

    end
    
        fea_FileName = strcat(file_name,'.mat');
        
    X = reshape(spec_mel_dB,size(spec_mel_dB,1),size(spec_mel_dB,2),2,[]);
    save(fullfile(Results_dir_Mel,fea_FileName), 'X');

    X = reshape(spec_bark_dB,size(spec_bark_dB,1),size(spec_bark_dB,2),2,[]);
    save(fullfile(Results_dir_Bark,fea_FileName), 'X');
    
    X = reshape(spec_gammatone_dB,size(spec_gammatone_dB,1),size(spec_gammatone_dB,2),2,[]);  % -
    save(fullfile(Results_dir_Gammatone,fea_FileName), 'X');
       
end


function []=CQT_based_TFRs_extration(inData,fs,file_name,Results_save_path)

    maxFreq = fs/2;
    minFreq = maxFreq/2^9;

    LPS_CQT_audio=zeros(256,213,2); 
    for i=1:size(inData,2)

        [LPS_segment,~]=cqt(inData(:,i),...
                 'SamplingFrequency',fs,'BinsPerOctave',96,'FrequencyLimits',[minFreq maxFreq]);

        LPS_CQT=abs(LPS_segment.c);
        LPS_CQT=(LPS_CQT).^2;
        spec = 20*log10(LPS_CQT+eps);


        spec=imresize(spec,[256 213]);

        
        LPS_CQT_audio(:,:,i)=spec;
    end
    X = reshape(LPS_CQT_audio,size(LPS_CQT_audio,1),size(LPS_CQT_audio,2),2,[]);

    fea_FileName = strcat(file_name,'.mat');
    save(fullfile(Results_save_path,fea_FileName), 'X');

end

function []=Scattering_extration(inData,fs,file_name,Results_save_path)
%%
    Results_dir_Scattering=fullfile(Results_save_path,'Scattering');

    if ~exist(Results_dir_Scattering,'dir')==1
         mkdir(Results_dir_Scattering);         
    end
    
    sf = waveletScattering('SignalLength',size(inData,1), ...
                       'SamplingFrequency',fs, ...
                       'InvarianceScale',0.75, ...
                       'QualityFactors',[4 1]);

    coeff1=featureMatrix(sf,inData(:,1),'Transform','log');
    coeff2=featureMatrix(sf,inData(:,2),'Transform','log');
    
    scatteringCoeffients(:,:,1)=coeff1;
    scatteringCoeffients(:,:,2)=coeff2;
                   
   fea_FileName = strcat(file_name,'.mat');
    
    X = reshape(scatteringCoeffients,size(scatteringCoeffients,1),size(scatteringCoeffients,2),2,[]);
    save(fullfile(Results_dir_Scattering,fea_FileName), 'X');
    
end

 
function []=CWT_based_TFRs_extration(inData,fs,file_name,Results_save_path)
%%
    Results_dir_Morse=fullfile(Results_save_path,'Morse');
    Results_dir_Amor=fullfile(Results_save_path,'amor');
    Results_dir_Bump=fullfile(Results_save_path,'bump');
    if ~exist(Results_dir_Morse,'dir')==1
         mkdir(Results_dir_Morse);         
    end
    if ~exist(Results_dir_Amor,'dir')==1
         mkdir(Results_dir_Amor);         
    end
    if ~exist(Results_dir_Bump,'dir')==1
         mkdir(Results_dir_Bump);         
    end
    % -------

    CWT_Morse=zeros(256,213,2); 
    CWT_Amor=zeros(256,213,2);
    CWT_Bump=zeros(256,213,2);
    % -Example: 'Wavelet','Morse', 'amor', or 'bump'
    fb_Morse = cwtfilterbank('SignalLength',size(inData,1),'SamplingFrequency',fs,'Wavelet','Morse','VoicesPerOctave',18);  % -默认是10
    fb_Amor = cwtfilterbank('SignalLength',size(inData,1),'SamplingFrequency',fs,'Wavelet','amor','VoicesPerOctave',18);
    fb_Bump = cwtfilterbank('SignalLength',size(inData,1),'SamplingFrequency',fs,'Wavelet','bump','VoicesPerOctave',18);
    
    inData = gpuArray(single(inData));
    
    
    for i=1:2

        cfs_Morse=gather(abs(fb_Morse.wt(inData(:,i))));
        cfs_Amor=gather(abs(fb_Amor.wt(inData(:,i))));
        cfs_Bump=gather(abs(fb_Bump.wt(inData(:,i))));
        
        
        spec_Morse = 20*log10(cfs_Morse+eps);
        spec_Amor = 20*log10(cfs_Amor+eps);
        spec_Bump = 20*log10(cfs_Bump+eps);
        
        
  
        spec_Morse=imresize(spec_Morse,[256 213]);
        spec_Amor=imresize(spec_Amor,[256 213]);
        spec_Bump=imresize(spec_Bump,[256 213]);

        CWT_Morse(:,:,i)=spec_Morse;
        CWT_Amor(:,:,i)=spec_Amor;
        CWT_Bump(:,:,i)=spec_Bump;
    end
    fea_FileName = strcat(file_name,'.mat');
    
    
    X = reshape(CWT_Morse,size(CWT_Morse,1),size(CWT_Morse,2),2,[]);
    save(fullfile(Results_dir_Morse,fea_FileName), 'X');
    
    X = reshape(CWT_Amor,size(CWT_Amor,1),size(CWT_Amor,2),2,[]);
    save(fullfile(Results_dir_Amor,fea_FileName), 'X');
    
    X = reshape(CWT_Bump,size(CWT_Bump,1),size(CWT_Bump,2),2,[]);
    save(fullfile(Results_dir_Bump,fea_FileName), 'X');
       
end




































