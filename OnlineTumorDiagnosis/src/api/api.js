import axios from '@/libs/api.request'

export function fileUpload(fileobj) {
    let param = new FormData()
    param.append('photo', fileobj.file)
    return axios.request({
        method: 'post',
        url: 'up_photo',
        headers: { 'Content-Type': 'multipart/form-data' },
        data: param
    })
}