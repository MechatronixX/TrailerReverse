images = cell(305,1);
for i = 1:305
    images{i} = imread(strcat('Truck_with_two_trailers',num2str(i+195),'.png'));
end
 % create the video writer with 1 fps
 writerObj = VideoWriter('myVideo.avi');
 writerObj.FrameRate = 26;
 % open the video writer
 open(writerObj);
 % write the frames to the video
 for u = 1:305
     % convert the image to a frame
     frame = im2frame(images{u});
     writeVideo(writerObj, frame);
 end
 % close the writer object
 close(writerObj);