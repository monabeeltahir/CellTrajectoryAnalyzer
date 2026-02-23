from trajectoryplot import TrajctoryPlot
from defanalysis import DefAnalysis_Dynamic

# === Run the function ===
if __name__ == "__main__":
    #videofilename = "No Magnetic Beads Cells 5uL_min.mp4"
    datapath = ['C:/Users/mt1102/Box/Nabeel Tahir Meetings/Meetings/Weekly Meetings/Meeting October 2025/3um Magnetic/', 'C:/Users/mt1102/Box/Nabeel Tahir Meetings/Meetings/Weekly Meetings/Meeting October 2025/3um Polystyrene/']
    videofillist = ["Run 2.mp4"]
    #videofillist = ["S1 V1 2uL.mp4","S1 V1 3uL.mp4", "S1 V2 3uL.mp4", "S1 V2 4uL.mp4", "S1 V2 5uL.mp4"]
    for i in range(2):
        print ("Running Path: ",  datapath[i])
    
        for  videofilename in videofillist:
            print("Running File: ", videofilename)
            videopath = datapath[i]+videofilename

            # TrajctoryPlot(min_frames=150, FileName = videopath[:-4]+"/"+"id_tracking_trajectories.csv", FolderName=videopath[:-4], gray_image_path=videopath[:-4]+"/GrayImage.png", 
            # do_tilt_correction=True,   save_corrected_plot=True,  # IMPORTANT
            # tilt_method="scharr_ransac",
            # tilt_roi="both",              # for "sobel": "top"/"bottom"; for ridge_mode: "top"/"bottom"/"both"
            # tilt_mode="both_midline",        # used only when tilt_method="ridge_mode"
            # )
    for vids in videofillist:
        contrlvidpath = datapath[1]+vids 
        expervidpath = datapath[0]+vids
        print(contrlvidpath)
        #df = pd.read_csv(expervidpath[:-4]+"/"+"filtered_trajectories.csv")
        #print(df.columns)
        DefAnalysis_Dynamic(ExperimentFile= expervidpath[:-4]+"/"+"filtered_trajectories_corrected.csv",  
                            ControlFile = contrlvidpath[:-4]+"/"+"filtered_trajectories_corrected.csv", OutputFolder=expervidpath[:-4], 
                            min_track_length = 40, sensitivity=99, correct_baseline_drift=False, fit_exponential_exp=True,
                    exp_min_x_span_px=40, CntrlOutputFolder= contrlvidpath[:-4])

