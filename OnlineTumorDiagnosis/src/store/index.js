import Vue from 'vue'
import Vuex from 'vuex'

Vue.use(Vuex)

const store = new Vuex.Store({
    state: {
        userInfo: {

        },
        shareUserInfo: {

        }
    },
    mutations: {
        SET_USER(state, payload) {
            state.userInfo = payload
        },
        SET_SHARE_USER(state, payload) {
            state.shareUserInfo = payload
        }
    },
    actions: {
        setUser({ commit }, payload) {
            commit('SET_USER', payload)
        },
        setShareUser({ commit }, payload) {
            commit('SET_SHARE_USER', payload)
        }
    },
    strict: false
})

export default store