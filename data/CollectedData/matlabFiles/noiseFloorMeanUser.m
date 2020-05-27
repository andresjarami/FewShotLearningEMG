function [noiseFloorEmg]=noiseFloorMeanUser(user)
load(strcat('all\',user,'\training\userData.mat'));
%user='AndresJaramillo';
for k=1:1:10
    % clear emgDataDcComponent;
    %     clear emgData;
    %     clear emgDataAUX;
    %     clear c;
    
    %     emgData=userDataTrain.gestures.relax.data{k,:}.emg;
    % meanSensor=mean(emgData,1);
    % for j=1:1:8
    %     emgDataDcComponent(:,j)=emgData(:,j)-meanSensor(j);
    % end
    
    % E=sum(C)/length(emgData);
    
    %     emgDataAUX=sum(transpose(abs(emgData)));
    %     samples=1;
    %     j=1;
    %     for i=1:samples:length(emgDataAUX)-mod(length(emgDataAUX),samples)
    %         c(j)=(emgDataAUX(i));
    %         j=j+1;
    %     end
    
    noiseFloor(k)=mean(sum(transpose(abs(...
        userDataTrain.gestures.relax.data{k,:}.emg))));
end

noiseFloorEmg=max(noiseFloor);
end