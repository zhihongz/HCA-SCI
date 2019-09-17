function Pic2Video(dn, picformat,disp_s, saveflag)
% CreatVideoFromPic(dn, picformat,aviname)
% 将某个文件夹下某种格式的所有图片合成为视频文件
% dn : 存储图片的文件夹
% picformat : 要读取的图片的格式，如png、jpg等形式，字符串数组
% disp_s   : 播放时两帧间的间隔时间 秒
% saveflag :存储视频标志
% example : CreatVideoFromPic( './', 'png','presentation.avi');

    if ~exist(dn, 'dir')
        error('dir not exist!!!!');
    end
    picname=fullfile( dn, strcat('*.',picformat));
    picname=dir(picname);
    aviname = 'result.avi';
    
    if saveflag
        aviobj = VideoWriter(aviname);
        open(aviobj);
    end

    for i=1:length(picname)
        picdata=imread( fullfile(dn, (picname(i,1).name)));
        imshow(picdata)
        text(10, 10, ['frame:',num2str(i)], 'Color', 'white', 'Fontsize', 10);
        pause(disp_s)		
        if saveflag && (~isempty( aviobj.Height))
            if size(picdata,1) ~= aviobj.Height || size(picdata,2) ~= aviobj.Width
                close(aviobj);
                delete( aviname )
                error('所有图片的尺寸要相同！！');
            end
        end
        
        if saveflag
            writeVideo(aviobj,picdata);
        end
    end
    if saveflag
        close(aviobj);
    end
end
