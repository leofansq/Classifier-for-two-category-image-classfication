function tree = make_tree(X,Y,base)
%�ݹ�����
[train_features, train_num] = size(X);
%ѵ��������Ӧ��������ظ���ȡȡ��𣬴Ӷ��õ���������
Y_uniqued = unique(Y);
%��ʼ�����ķ�������Ϊ��0��
tree.dim = 0;
%��ʼ������λ����inf 
tree.split_loc = inf;

%��ѵ�����������ֻʣһ��ʱ���������֦ ֹͣ
if ((train_num == 1) || (length(Y_uniqued) == 1))
    %ͳ��������𼰷ֱ�����ÿ������������Ŀ
    H = hist(Y, length(Y_uniqued));%http://www.ilovematlab.cn/thread-297325-1-1.html
    %�õ����������������Ǹ�����λ��largest
    [~, largest] 	= max(H);
    tree.Nf         = [];
    tree.split_loc  = [];
    %��ʱ���ذ������������һ����Ϊ�����
    tree.child	 	= Y_uniqued(largest);    
    return
end


%����������Ŀ&�������е���Ϣ�� 
for i = 1:1:length(Y_uniqued)
    %�õ���ǰ�������������Ϊ��i����������ռ���������ı���  
    Pnode(i) = length(find(Y == Y_uniqued(i))) / train_num;
end
%���㵱ǰ����Ϣ��
Inode = -sum(Pnode.*log(Pnode)/log(2));

%����������ÿ�������ֱ������Ϣ��  
%��¼ÿ����������Ϣ���� ��ʼ��Ϊ0
delta_Ib    = zeros(1, train_features);
%��ʼ��ÿ�������ķ���ֵ��inf
split_loc	= ones(1, train_features)*inf;

for i = 1:train_features
    %��ǰ����ѵ��������ĳ�������ľ���ֵ
    data = X(i,:);
    %���ظ��ľ�������ֵ 
    data_uniqued = unique(data);
    %���ظ��ľ���ֵ��Ŀ
    data_uniqued_len = length(data_uniqued);
    
    if (data_uniqued_len==1)
        delta_Ib(i)=-2000;
        continue;
    end
    %��һ�д洢����ǰn������λ�õ������еı�ǩ�ֲ�������ڶ��д����ǰn������λ�õ�����֮��ı�ǩ�ֲ����
     P	= zeros(length(Y_uniqued), 2);
     %��С��������ĳ������ֵ�ľ���ֵ
     %[sA,index] = sort(A) �������sA������õ�������index �� ����sA �ж� A ������
     [sorted_data, indices] = sort(data);
     %���б����������˳�����������
     sorted_class = Y(indices);
     
     %���������Ϣ����
     I	= zeros(1, data_uniqued_len - 1);
     temp_split=zeros(1,data_uniqued_len - 1);
     for j = 1:data_uniqued_len-1
         temp=(data_uniqued(j)+data_uniqued(j+1))/2;
         temp_split(j)=temp;
         i1=find(sorted_data<temp);
         ii=length(i1);
         for k =1:length(Y_uniqued)
             %��¼<=��ǰ����ֵ�����������ķֲ����
             P(k,1) = length(find(sorted_class(1:ii) == Y_uniqued(k)));
             %��¼>��ǰ����ֵ�����������ķֲ����
             P(k,2) = length(find(sorted_class(ii+1:end) == Y_uniqued(k)));
         end    
         %����λ��ǰ�����������&����
         Ps  = sum(P);
         P   = P./(eps+repmat(Ps,size(P,1),1));  
         Ps  = Ps/sum(Ps); 
         %������Ϣ��
         info = sum(-P.*log(eps+P));
         %��j���������ѵĵ���Ϣ���� 
         I(j) = (Inode - sum(info.*Ps));  
     end
       %�ҵ���Ϣ�������Ļ��ַ���
       %delta_Ib(i)�д�ŵ��Ƕ��ڵ�ǰ��i���������ԣ�������Ϣ������Ϊ�����������Ϣ����  s���������ַ���
        [delta_Ib(i), s] = max(I);
        %��Ӧ����i�Ļ���λ�þ�����ʹ��Ϣ�������Ļ���ֵ
        split_loc(i) = temp_split(s);
end
    
if delta_Ib==ones(1,train_features)*-2000
    H				= hist(Y, length(Y_uniqued));
    [~, largest] 	= max(H);
    tree.dim        =0;
    tree.Nf         = [];
    tree.split_loc  = [];
    tree.child	 	= Y_uniqued(largest);
    return
end

%�ҵ���ǰҪ��Ϊ��������������
%�ҵ�����������������Ϣ�����Ӧ������
[~, dim]    = max(delta_Ib);
dims        = 1:train_features;
%��Ϊ���ķ�������
tree.dim    = dim;
%�õ���ǰ�����������������������ֵ
Nf	= unique(X(dim,:));
data_uniqued_len = length(Nf);
%��Ϊ���ķ�����������
tree.Nf = Nf;
%����������Ļ���λ�ü�Ϊ���ķ���λ��
tree.split_loc = split_loc(dim);

%��ֻʣ��һ������ʱ����
if (data_uniqued_len == 1)
    %ͳ�Ƶ�ǰ������������𼰷ֱ�����ÿ������������Ŀ
    H = hist(Y, length(Y_uniqued));
    [~, largest] 	= max(H);
    tree.dim        =0;
    tree.Nf         = [];
    tree.split_loc  = [];
    tree.child	 	= Y_uniqued(largest);
    return
end

%�ҵ�����ֵ<=����ֵ������������
indices1 = find(X(dim,:) <= split_loc(dim));
%�ҵ�����ֵ>����ֵ������������
indices2 = find(X(dim,:) > split_loc(dim));
%���<=����ֵ >����ֵ��������Ŀ��������0����������
if ~(isempty(indices1) || isempty(indices2))
    tree.child(1) = make_tree(X(dims, indices1), Y(indices1),base+1);
    tree.child(2) = make_tree(X(dims, indices2), Y(indices2),base+1);
else
    
    H = hist(Y, length(Y_uniqued));
    [~, largest] = max(H);
    tree.child = Y_uniqued(largest);
    %���ķ���������Ϊ0
    tree.dim = 0;
end