function save_ICA_plots(EEG, dirname, suffix, extension)
    %% Save the overall component map and each map of the individual components
    pt_name_split = split(EEG.filename, ".");
    pt_name = string(pt_name_split(1));
    component_map_fname = pt_name + '_component' + suffix;  

    n_components = size(EEG.icawinv, 2);
    f1 = pop_selectcomps(EEG, [1:n_components] );  % Plots the overall components
    
    component_map_fpath = fullfile(dirname, component_map_fname);
    saveas(gcf, component_map_fpath, extension);
    for comp = 1:n_components
        pop_prop( EEG, 0, comp, 0, {'freqrange',[1 50] });  % Plots the specified component
        component_map_fname = pt_name + '_component_'+string(comp) + suffix;
        component_map_fpath = fullfile(dirname, component_map_fname);
        saveas(gcf, component_map_fpath, extension);
        close all
    end
end