#include <iostream>
#include <string>
#include <vector>
using namespace std;


struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode() : val(0), left(nullptr), right(nullptr) {}
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
    TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
};
 
void inorder(vector<int>& v,TreeNode* root)  //I don't understand the reccurence
{
    if(root==NULL)
        return ;
    inorder(v,root->left);
    v.push_back(root->val);
    inorder(v,root->right);
}

int main(){
    TreeNode *root,*b,*c,*d;
    d = new TreeNode(1);
    c = new TreeNode(2,nullptr,d);
    b = new TreeNode(3,c,nullptr);
    root = new TreeNode(4,b,nullptr);
    
    
    
    vector<int> v;
    inorder(v,root);
    // return v;
    for(int i:v){
        cout<<i<<" ";
    }
    
    
    delete(root);
    delete(b);
    delete(c);
    delete(d);
}
