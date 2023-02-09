close all; clear all;

headerFile = {
    '../eeg_raw/SSVEP_noMR_1.vhdr';
    };

markerFile={
    '../eeg_raw/SSVEP_noMR_1.vmrk';
    };

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EEG setup
%
n_chan=32; %32-channel EEG;
select_channel={ 'Fp1'    'Fp2'    'F3'    'F4'    'C3'    'C4'    'P3'    'P4'    'O1'    'O2'    'F7'    'F8'    'T7'    'T8'    'P7'    'P8'    'Fz'    'Cz'    'Pz'    'Oz'    'FC1'    'FC2'    'CP1'    'CP2'    'FC5'    'FC6'    'CP5'    'CP6'    'TP9'    'TP10'    'POz'    'ECG'};
eeg_channel=[1:31];
ecg_channel=[32];

%these two tokens are required (but values are arbitrary) for data
%collected inside MRI
trigger_token=1e3;
sync_token=1e2;

%output_file='erp.mat';
output_avg_file='erp_avg_inside.mat';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% start processing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for f_idx=1:length(headerFile)
    [dummy,fstem]=fileparts(headerFile{f_idx});
    fprintf('reading [%s]...\n',fstem);
    
    % first get the continuous data as a matlab array
    eeg{f_idx} = double(bva_loadeeg(headerFile{f_idx}));
    
    % meta information such as samplingRate (fs), labels, etc
    [fs(f_idx) label meta] = bva_readheader(headerFile{f_idx});
    
    found_channel={};
    for s_idx=1:length(select_channel)
        IndexC = strcmp(lower(label),lower(select_channel{s_idx})); %change all labels into lower case
        Index = find(IndexC);
        
        if(~isempty(Index))
            fprintf('\tChannel [%s] found:: index=%03d \r',select_channel{s_idx},Index);
            if(strcmp(lower(select_channel{s_idx}),'ecg'))
                ecg_channel=Index;
            else
                eeg_channel(s_idx)=Index;
            end;
            found_channel{end+1}=select_channel{s_idx};
        else
            fprintf('\tChannel [%s] not found! \r',select_channel{s_idx});
        end;
    end;
    fprintf('\n');
    
    ecg{f_idx}=eeg{f_idx}(ecg_channel,:)';
    eeg{f_idx}=eeg{f_idx}(eeg_channel,:);
end;