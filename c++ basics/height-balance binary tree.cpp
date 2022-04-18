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
 
TreeNode* inputnode(vector<int>nums, int left, int right){
    if(left>right){
        return NULL;
    }
    int mid = (right+left)/2;
    int val = nums[mid];
    TreeNode* root = new TreeNode(val);
    root->left = inputnode(nums,left,mid-1);
    root->right = inputnode(nums,mid+1,right);
    return root;
}
int main(){
    vector<int>nums = {0,1,2,3,4,5,6};
    int n = nums.size()-1;
    inputnode(nums,0,n);

}
