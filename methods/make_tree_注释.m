function tree = make_tree(X,Y,base)
%递归求树
[train_features, train_num] = size(X);
%训练样本对应的类别（无重复的取取类别，从而得到类别个数）
Y_uniqued = unique(Y);
%初始化树的分裂特征为第0个
tree.dim = 0;
%初始化分裂位置是inf 
tree.split_loc = inf;

%当训练样本或类别只剩一个时无需继续分枝 停止
if ((train_num == 1) || (length(Y_uniqued) == 1))
    %统计样本类别及分别属于每个类别的样本数目
    H = hist(Y, length(Y_uniqued));%http://www.ilovematlab.cn/thread-297325-1-1.html
    %得到包含样本数最多的那个类别的位置largest
    [~, largest] 	= max(H);
    tree.Nf         = [];
    tree.split_loc  = [];
    %暂时返回包含样本数最多一类作为其类别
    tree.child	 	= Y_uniqued(largest);    
    return
end


%遍历类别的数目&计算现有的信息量 
for i = 1:1:length(Y_uniqued)
    %得到当前所有样本中类别为第i个类别的样本占样本总数的比例  
    Pnode(i) = length(find(Y == Y_uniqued(i))) / train_num;
end
%计算当前的信息熵
Inode = -sum(Pnode.*log(Pnode)/log(2));

%对特征集中每个特征分别计算信息熵  
%记录每个特征的信息增益 初始化为0
delta_Ib    = zeros(1, train_features);
%初始化每个特征的分裂值是inf
split_loc	= ones(1, train_features)*inf;

for i = 1:train_features
    %当前所有训练样本的某个特征的具体值
    data = X(i,:);
    %无重复的具体特征值 
    data_uniqued = unique(data);
    %无重复的具体值数目
    data_uniqued_len = length(data_uniqued);
    
    if (data_uniqued_len==1)
        delta_Ib(i)=-2000;
        continue;
    end
    %第一列存储代表前n个分裂位置的样本中的标签分布情况，第二列代表除前n个分裂位置的样本之外的标签分布情况
     P	= zeros(length(Y_uniqued), 2);
     %从小到大排序某个特征值的具体值
     %[sA,index] = sort(A) ，排序后，sA是排序好的向量，index 是 向量sA 中对 A 的索引
     [sorted_data, indices] = sort(data);
     %将判别类别随样本顺序调整而调整
     sorted_class = Y(indices);
     
     %计算分裂信息度量
     I	= zeros(1, data_uniqued_len - 1);
     temp_split=zeros(1,data_uniqued_len - 1);
     for j = 1:data_uniqued_len-1
         temp=(data_uniqued(j)+data_uniqued(j+1))/2;
         temp_split(j)=temp;
         i1=find(sorted_data<temp);
         ii=length(i1);
         for k =1:length(Y_uniqued)
             %记录<=当前特征值的样本的类别的分布情况
             P(k,1) = length(find(sorted_class(1:ii) == Y_uniqued(k)));
             %记录>当前特征值的样本的类别的分布情况
             P(k,2) = length(find(sorted_class(ii+1:end) == Y_uniqued(k)));
         end    
         %分裂位置前后各有样本数&比例
         Ps  = sum(P);
         P   = P./(eps+repmat(Ps,size(P,1),1));  
         Ps  = Ps/sum(Ps); 
         %计算信息熵
         info = sum(-P.*log(eps+P));
         %第j个样本分裂的的信息增益 
         I(j) = (Inode - sum(info.*Ps));  
     end
       %找到信息增益最大的划分方法
       %delta_Ib(i)中存放的是对于当前第i个特征而言，最大的信息增益作为这个特征的信息增益  s存放这个划分方法
        [delta_Ib(i), s] = max(I);
        %对应特征i的划分位置就是能使信息增益最大的划分值
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

%找到当前要作为分裂特征的特征
%找到所有特征中最大的信息增益对应的特征
[~, dim]    = max(delta_Ib);
dims        = 1:train_features;
%记为树的分裂特征
tree.dim    = dim;
%得到当前所有样本的这个特征的特征值
Nf	= unique(X(dim,:));
data_uniqued_len = length(Nf);
%记为树的分类特征向量
tree.Nf = Nf;
%把这个特征的划分位置记为树的分裂位置
tree.split_loc = split_loc(dim);

%当只剩下一个特征时结束
if (data_uniqued_len == 1)
    %统计当前所有样本的类别及分别属于每个类别的样本数目
    H = hist(Y, length(Y_uniqued));
    [~, largest] 	= max(H);
    tree.dim        =0;
    tree.Nf         = [];
    tree.split_loc  = [];
    tree.child	 	= Y_uniqued(largest);
    return
end

%找到特征值<=分裂值的样本的索引
indices1 = find(X(dim,:) <= split_loc(dim));
%找到特征值>分裂值的样本的索引
indices2 = find(X(dim,:) > split_loc(dim));
%如果<=分裂值 >分裂值的样本数目都不等于0，继续分裂
if ~(isempty(indices1) || isempty(indices2))
    tree.child(1) = make_tree(X(dims, indices1), Y(indices1),base+1);
    tree.child(2) = make_tree(X(dims, indices2), Y(indices2),base+1);
else
    
    H = hist(Y, length(Y_uniqued));
    [~, largest] = max(H);
    tree.child = Y_uniqued(largest);
    %树的分裂特征记为0
    tree.dim = 0;
end