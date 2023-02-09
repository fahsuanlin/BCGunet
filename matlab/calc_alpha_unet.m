close all; clear all;

root_path='/Users/fhlin/workspace/eegmri_openclose';

fn=dir(sprintf('%s/noscan_bcgunet/*.mat',root_path));

EEG_obs_open=[];
EEG_obs_close=[];
EEG_unet_open=[];
EEG_unet_close=[];


for f_idx=1:length(fn)
    fprintf('loading [%s]...\n',fn(f_idx).name);
    newStr = split(fn(f_idx).name,'-');
    f_obs=sprintf('%s/%s/analysis/%s',root_path,newStr{1},newStr{2});
    load(f_obs);
    if(~isempty(findstr(lower(newStr{2}),'open')))
        EEG_obs_open(:,end+1)=mean(abs(inverse_waveletcoef(10,double(EEG),sfreq,5)),2);

        load(sprintf('%s/noscan_bcgunet/%s',root_path,fn(f_idx).name));
        EEG_unet_open(:,end+1)=mean(abs(inverse_waveletcoef(10,double(EEG_removeBCG_unet),sfreq,5)),2);
    else
        EEG_obs_close(:,end+1)=mean(abs(inverse_waveletcoef(10,double(EEG),sfreq,5)),2);

        load(sprintf('%s/noscan_bcgunet/%s',root_path,fn(f_idx).name));
        EEG_unet_close(:,end+1)=mean(abs(inverse_waveletcoef(10,double(EEG_removeBCG_unet),sfreq,5)),2);
    end;
end;

save calc_alpha_unet.mat sfreq EEG_unet_open EEG_unet_close EEG_obs_close EEG_obs_open fn


load bem.mat;
verts_osc_electrode_idx(end-2:end,:)=[];
figure;
etc_render_topo('vol_vertex',verts_osc,'vol_face',faces_osc-1,'topo_vertex',verts_osc_electrode_idx-1,'topo_stc',mean(EEG_obs_close,2)./mean(EEG_obs_open,2),'topo_smooth',10,'topo_threshold',[1.25 1.5],'topo_stc_timevec_unit','Hz','view_angle',[0 50]);
figure;
etc_render_topo('vol_vertex',verts_osc,'vol_face',faces_osc-1,'topo_vertex',verts_osc_electrode_idx-1,'topo_stc',mean(EEG_unet_close,2)./mean(EEG_unet_open,2),'topo_smooth',10,'topo_threshold',[1.25 1.5],'topo_stc_timevec_unit','Hz','view_angle',[0 50]);
