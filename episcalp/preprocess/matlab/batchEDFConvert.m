function batchEDFConvert(inputFolder, outputFolder)
edf_files = dir(inputFolder + "/*.edf");
for i = 1:size(edf_files,1)
    fname = edf_files(i).name;
    fpath = inputFolder + "/" + fname;
    [hdr, record] = edfread(fpath);
    output_fname = strrep(fname, "edf", "mat");
    output_fpath = outputFolder + "/" + output_fname;
    save(output_fpath, "hdr", "record")
end