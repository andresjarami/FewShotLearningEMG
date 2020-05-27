function [emgDetectedData,initialTime,finalTime]=gestureDetection(emgData,Tol,plotT)

j=4;
wi=j;

overlappingInitial=1;

securityIntervalInitial=40+j;

portionEMG=200;

initialTime=0;
finalTime=0;
interval=0;


D=sum(transpose(abs(emgData)));
aux=0;

while ((initialTime==0)&&(j<length(emgData)))
    
    if D(j)>Tol
        aux=aux+1;
    else
        aux=0;
    end
    
    for g=1:1:wi
        vectorTestInitial(g)=D(j-g+1);
    end
    
    if sum(vectorTestInitial)>(Tol*wi)
        initialTime=j-securityIntervalInitial;
        j=length(emgData);
    end
    j=j+overlappingInitial;
end
finalTime=initialTime+portionEMG;

if initialTime<0
    initialTime=1;
end
if finalTime>length(emgData)
    finalTime=length(emgData);
end

if (initialTime~=0) && (finalTime~=0)
    for k=initialTime:1:finalTime
        emgDetectedData(k-initialTime+1,:)=emgData(k,:);
    end
else
    emgDetectedData=0;
    initialTime=0;
    finalTime=0;
end

if plotT==1
    plotEmgDetec(emgData,initialTime,finalTime)
end

end