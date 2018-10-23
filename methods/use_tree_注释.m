function class = use_tree(X_test, index, tree)
%设置每个样本的初始预测标签都是0
class = zeros(1, size(X_test,2));
%到达树末端
if (tree.dim == 0)
    %得到样本对应的标签是tree.child
    class(index) = tree.child;
    return
end
%未到达树末端
%得到分裂特征及特征的索引
dim = tree.dim;
dims= 1:size(X_test,1);
%找到当前测试样本中这个特征的特征值<=分裂值的样本索引 
in = index(find(X_test(dim, index) <= tree.split_loc));
%对这部分样本再分叉
class	= class + use_tree(X_test(dims, :), in, tree.child(1));
%找到当前测试样本中这个特征的特征值>分裂值的样本索引
in = index(find(X_test(dim, index) >  tree.split_loc));
%对这部分样本再分叉
class = class + use_tree(X_test(dims, :), in, tree.child(2));