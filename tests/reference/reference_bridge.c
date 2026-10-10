// Validation only. reference_run.c is the upstream MIT-licensed llama2.c source.
// Kept outside the teaching project and never linked into mini_infer_main.
#define TESTING
#include "reference_run.c"

void* ref_model_open(const char* path) {
    Transformer* t = malloc(sizeof(Transformer));
    build_transformer(t,(char*)path);
    return t;
}
const float* ref_forward(void* model,int token,int position) {
    return forward((Transformer*)model,token,position);
}
void ref_model_close(void* model) {free_transformer(model);free(model);}
void* ref_tokenizer_open(const char* path,int vocab) {
    Tokenizer* t=malloc(sizeof(Tokenizer));
    build_tokenizer(t,(char*)path,vocab);
    return t;
}
int ref_encode(void* tokenizer,const char* text,int* tokens) {
    int n=0;
    encode(tokenizer,(char*)text,1,0,tokens,&n);
    return n;
}
void ref_tokenizer_close(void* tokenizer) {free_tokenizer(tokenizer);free(tokenizer);}
