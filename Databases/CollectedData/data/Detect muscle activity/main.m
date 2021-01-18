

clear all

for i=1:1:60
    
    user=join(['user',num2str(i)]);    
    
    j=1;
    for dataType=0:1
       
        if dataType==0
            load(strcat('allUsers\',user,'\training\userData.mat'));
            userData=userDataTrain;
        elseif dataType==1
            load(strcat('allUsers\',user,'\testing\userData.mat'));
            userData=userDataTest;
        end
        
        
        
        for gestureAUX=1:1:5
            
            for numberGesture=1:1:25
                
                if gestureAUX==1
                    emg=segmentationGesture(userData.gestures.fist.data{numberGesture,:}.emg);
                    
                    save(strcat('detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');                    
                    
                    
                elseif gestureAUX==2
                    emg=segmentationGesture(userData.gestures.waveIn.data{numberGesture,:}.emg);
                    
                    
                    save(strcat('detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                elseif gestureAUX==3
                    emg=segmentationGesture(userData.gestures.waveOut.data{numberGesture,:}.emg);
              
                    save(strcat('detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                elseif gestureAUX==4
                    emg=segmentationGesture(userData.gestures.fingersSpread.data{numberGesture,:}.emg);
                 
                   save(strcat('detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                elseif gestureAUX==5
                    emg=segmentationGesture(userData.gestures.doubleTap.data{numberGesture,:}.emg);
               
                    save(strcat('detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                end
            end
        end
       
    end
    
end

