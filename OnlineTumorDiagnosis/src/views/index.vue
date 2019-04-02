<template>
  <div>
    <el-menu
      :default-active="activeIndex"
      class="el-menu-demo"
      mode="horizontal"
      @select="handleSelect"
    >
      <el-menu-item index="1"><h2>在线肿瘤诊断系统</h2></el-menu-item>
    </el-menu>
    <el-row class="imgup">
      <el-col :span='24'>
        <el-row>
            <el-col :span="10">
        <div>
          <el-row>
            <el-col :span="12">
              <el-upload class="upload-demo" action :http-request="customUpload" :limit="1" :show-file-list='false'>
                <el-button size="small" type="primary">上传DMC</el-button>
              </el-upload>
            </el-col>
            <el-col :span="12">
              <el-button size="small" type="primary" @click='start'>开始诊断</el-button>
            </el-col>
          </el-row>
          <el-row>
            <el-col :span="24">
              <img :src="uploadpic.file_path" alt='CT影像' class="CT" v-if="uploadpic.file_path != ''">
              <img src="../assets/default.png" alt='CT影像' class="CT" v-if="uploadpic.file_path == ''">
            </el-col>
          </el-row>
          <el-row>
            <el-col :span="24">
              <span>CT影像</span>
            </el-col>
          </el-row>
        </div>
      </el-col>
      <el-col :span="14">
        <div>
          <el-row>
            <el-col :span="24" class="text-left">
              <span >诊断结果:</span>
            </el-col>
          </el-row>
          <el-row>
            <el-col :span="24">
              <span>78%可能为恶性肿瘤</span>
            </el-col>
          </el-row>
        </div>
      </el-col>
        </el-row>
      </el-col>
    </el-row>
    <el-row>
      <el-col :span="24" class="text-left">
        <span >诊断依据:</span>
      </el-col>
    </el-row>
    <el-row>
            <el-col :span="6">
              <img src="http://127.0.0.1:8808/show/2019040211443890.png" alt>
            </el-col>
            <el-col :span="6">
              <img src="../assets/logo.png" alt>
            </el-col>
            <el-col :span="6">
              <img src="../assets/logo.png" alt>
            </el-col>
            <el-col :span="6">
              <img src="../assets/logo.png" alt>
            </el-col>
            <el-col :span="6">
              <img src="../assets/logo.png" alt>
            </el-col>
            <el-col :span="6">
              <img src="../assets/logo.png" alt>
            </el-col>
    </el-row>
  </div>
</template>
<script>
import {fileUpload} from '@/api/api'
export default {
  name: "index",
  data() {
    return {
      activeIndex: "1",
      uploadpic:{
        file_path:''
      }
    };
  },
  methods: {
      start(){
            console.log('开始诊断')
      },
    customUpload(file) {
      // this.generatorFileMd5(file.file)
      // 自定义上传
      fileUpload(file).then(response => {
        console.log(response)
        if(response){
          if(response.data.success == 0){
          this.uploadpic = response.data.data
          this.$message({
            message: response.data.msg,
            type: 'success'
          });
        }else{
          this.$message({
            message: response.data.msg,
            type: 'warning'
          });
        }
        }else{
          this.$message({
            message: '上传失败',
            type: 'warning'
          });
        }
      }).then(err =>{
        if(this.uploadpic.file_path == ''){
          this.$message({
            message: '上传失败',
            type: 'warning'
          });
        } 
      })
    },
    handleSelect(key, keyPath) {
      console.log(key, keyPath);
    }
  },
  mounted(){
    console.log(window.location.href)
  }
};
</script>
<style lang="scss" scoped>
h2{
    margin: 0
}
.text-left{
    text-align: left
}
.el-row {
  margin-bottom: 20px;
  width: 100%;
  &:last-child {
    margin-bottom: 0;
  }
}
.el-col {
  border-radius: 4px;
}
.bg-purple-dark {
  background: #99a9bf;
}
.bg-purple {
  background: #d3dce6;
}
.bg-purple-light {
  background: #e5e9f2;
}
.grid-content {
  border-radius: 4px;
  min-height: 36px;
}
.row-bg {
  padding: 10px 0;
  background-color: #f9fafc;
}
.CT{
  width: 50%;
}
</style>
