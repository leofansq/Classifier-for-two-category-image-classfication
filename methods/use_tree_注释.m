function class = use_tree(X_test, index, tree)
%����ÿ�������ĳ�ʼԤ���ǩ����0
class = zeros(1, size(X_test,2));
%������ĩ��
if (tree.dim == 0)
    %�õ�������Ӧ�ı�ǩ��tree.child
    class(index) = tree.child;
    return
end
%δ������ĩ��
%�õ���������������������
dim = tree.dim;
dims= 1:size(X_test,1);
%�ҵ���ǰ�����������������������ֵ<=����ֵ���������� 
in = index(find(X_test(dim, index) <= tree.split_loc));
%���ⲿ�������ٷֲ�
class	= class + use_tree(X_test(dims, :), in, tree.child(1));
%�ҵ���ǰ�����������������������ֵ>����ֵ����������
in = index(find(X_test(dim, index) >  tree.split_loc));
%���ⲿ�������ٷֲ�
class = class + use_tree(X_test(dims, :), in, tree.child(2));