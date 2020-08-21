

clear all

for i=1:1:60
    
    if i==1
        user='AlexandraGranda';
    elseif i==2
        user='AlexToasa';
    elseif i==3
        user='AnaMunhoz';
    elseif i==4
        user='AndreaSilva';
    elseif i==5
        user='AndresGarcia';
    elseif i==6
        user='AndresGuerra';
    elseif i==7
        user='AndresJaramillo';
    elseif i==8
        user='AnthonyAlmachi';
    elseif i==9
        user='BolivarRoman';
    elseif i==10
        user='CamiloPineda';
    elseif i==11
        user='CarlosChicaiza';
    elseif i==12
        user='ChristianFernandez';
    elseif i==13
        user='CynthiaAlvarez';
    elseif i==14
        user='DanielTacoGallardo';
    elseif i==15
        user='DavidDavila';
    elseif i==16
        user='DiegoMarquez';
    elseif i==17
        user='DiegoVelastegui';
    elseif i==18
        user='EdgarCahuenas';
    elseif i==19
        user='EdisonCabrera';
    elseif i==20
        user='EdwinEnriquez';
    elseif i==21
        user='ErickTipan';
    elseif i==22
        user='EstebanAndaluz';
    elseif i==23
        user='EstefanCevallos';
    elseif i==24
        user='EvelynRegalado';
    elseif i==25
        user='FrancisSoria';
    elseif i==26
        user='FrankAldana';
    elseif i==27
        user='FreddyNieto';
    elseif i==28
        user='GabrielaRamos';
    elseif i==29
        user='GeomaraFajardo';
    elseif i==30
        user='GracielaGonzalez';
    elseif i==31
        user='HomeroArias';
    elseif i==32
        user='JonathanAlarcon';
    elseif i==33
        user='JonathanRamos';
    elseif i==34
        user='JonathanZea';
    elseif i==35
        user='JoshuaRosero';
    elseif i==36
        user='JuanBalseca';
    elseif i==37
        user='JuanJoseMorales';
    elseif i==38
        user='JuanLopezR';
    elseif i==39
        user='KatherineVela';
    elseif i==40
        user='KevinMachado';
    elseif i==41
        user='LeninDavidMinho';
    elseif i==42
        user='LeslieLopez';
    elseif i==43
        user='LeslyTello';
    elseif i==44
        user='LuisAlmeida';
    elseif i==45
        user='LuisUnapanta';
    elseif i==46
        user='MarcoSegura';
    elseif i==47
        user='MarielaVasquez';
    elseif i==48
        user='NicoleOntaneda';
    elseif i==49
        user='OscarRivera';
    elseif i==50
        user='PaulGuala';
    elseif i==51
        user='PaulLora';
    elseif i==52
        user='PaulReinoso';
    elseif i==53
        user='RafaelVinueza';
    elseif i==54
        user='RenatoBurbano';
    elseif i==55
        user='RonaldAlvarado';
    elseif i==56
        user='SantiagoLema';
    elseif i==57
        user='SofiaGuerrero';
    elseif i==58
        user='WendyCalero';
    elseif i==59
        user='XavierCadena';
    elseif i==60
        user='XavierReinoso';
    else
        control=1;
    end
    
    j=1;
    for dataType=0:1
       
        if dataType==0
            load(strcat('all\',user,'\training\userData.mat'));
            userData=userDataTrain;
        elseif dataType==1
            load(strcat('all\',user,'\testing\userData.mat'));
            userData=userDataTest;
        end
        
        
        
        for gestureAUX=1:1:5
            
            for numberGesture=1:1:25
                
                if gestureAUX==1
                    emg=segmentationGesture(userData.gestures.fist.data{numberGesture,:}.emg);
                    
                    save(strcat('CollectedData\detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');                    
                    
                    
                elseif gestureAUX==2
                    emg=segmentationGesture(userData.gestures.waveIn.data{numberGesture,:}.emg);
                    
                    
                    save(strcat('CollectedData\detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                elseif gestureAUX==3
                    emg=segmentationGesture(userData.gestures.waveOut.data{numberGesture,:}.emg);
              
                    save(strcat('CollectedData\detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                elseif gestureAUX==4
                    emg=segmentationGesture(userData.gestures.fingersSpread.data{numberGesture,:}.emg);
                 
                   save(strcat('CollectedData\detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                elseif gestureAUX==5
                    emg=segmentationGesture(userData.gestures.doubleTap.data{numberGesture,:}.emg);
               
                    save(strcat('CollectedData\detectedData\emg_person',...
                        num2str(i),'_class',num2str(gestureAUX),'_rpt',num2str(numberGesture),...
                        '_type',num2str(dataType)),'emg');
                end
            end
        end
       
    end
    i
end

