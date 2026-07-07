/*
* 语音听写(iFly Auto Transform)技术能够实时地将语音转换成对应的文字。
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
#include <unistd.h>
#include <termio.h>
#include "qisr.h"
#include "msp_cmn.h"
#include "msp_errors.h"
#include "speech_recognizer.h"
#include <iconv.h>

#include "ros/ros.h"
#include "std_msgs/String.h"

#define FRAME_LEN   640 
#define BUFFER_SIZE 4096

int wakeupFlag   = 0 ;
int resultFlag   = 0 ;

static void show_result(char *string, char is_over)
{
    resultFlag=1;   
    printf("\rResult: [ %s ]", string);
    if(is_over)
        putchar('\n');
}

static char *g_result = NULL;
static unsigned int g_buffersize = BUFFER_SIZE;

static bool is_valid_utf8(const char *input)
{
    if (!input)
        return true;

    const unsigned char *bytes = (const unsigned char *)input;
    while (*bytes) {
        if (*bytes <= 0x7F) {
            bytes++;
        } else if ((*bytes & 0xE0) == 0xC0) {
            if ((bytes[1] & 0xC0) != 0x80)
                return false;
            bytes += 2;
        } else if ((*bytes & 0xF0) == 0xE0) {
            if ((bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80)
                return false;
            bytes += 3;
        } else if ((*bytes & 0xF8) == 0xF0) {
            if ((bytes[1] & 0xC0) != 0x80 || (bytes[2] & 0xC0) != 0x80 || (bytes[3] & 0xC0) != 0x80)
                return false;
            bytes += 4;
        } else {
            return false;
        }
    }
    return true;
}

static std::string convert_encoding(const char *input, const char *from_encoding, const char *to_encoding)
{
    if (!input)
        return "";

    iconv_t cd = iconv_open(to_encoding, from_encoding);
    if (cd == (iconv_t)-1)
        return input;

    size_t input_len = strlen(input);
    size_t output_len = input_len * 4 + 16;
    std::string output(output_len, '\0');

    char *in_buf = const_cast<char *>(input);
    char *out_buf = &output[0];
    size_t in_left = input_len;
    size_t out_left = output_len;

    size_t ret = iconv(cd, &in_buf, &in_left, &out_buf, &out_left);
    iconv_close(cd);

    if (ret == (size_t)-1)
        return input;

    output.resize(output_len - out_left);
    return output;
}

static std::string ensure_utf8(const char *input)
{
    if (!input)
        return "";
    if (is_valid_utf8(input))
        return input;
    return convert_encoding(input, "GB18030", "UTF-8");
}

void on_result(const char *result, char is_last)
{
    if (result) {
        size_t left = g_buffersize - 1 - strlen(g_result);
        size_t size = strlen(result);
        if (left < size) {
            g_result = (char*)realloc(g_result, g_buffersize + BUFFER_SIZE);
            if (g_result)
                g_buffersize += BUFFER_SIZE;
            else {
                printf("mem alloc failed\n");
                return;
            }
        }
        strncat(g_result, result, size);
        show_result(g_result, is_last);
    }
}

void on_speech_begin()
{
    if (g_result)
    {
        free(g_result);
    }
    g_result = (char*)malloc(BUFFER_SIZE);
    g_buffersize = BUFFER_SIZE;
    memset(g_result, 0, g_buffersize);

    printf("Start Listening...\n");
    printf("Press \"Space\" key Stop\n");
}
void on_speech_end(int reason)
{
    if (reason == END_REASON_VAD_DETECT)
        printf("\nSpeaking done \n");
    else
        printf("\nRecognizer error %d\n", reason);
}

/* demo recognize the audio from microphone */
static void demo_mic(const char* session_begin_params)
{
    int errcode;
    int i = 0;

    struct speech_rec iat;

    struct speech_rec_notifier recnotifier = {
        on_result,
        on_speech_begin,
        on_speech_end
    };

    errcode = sr_init(&iat, session_begin_params, SR_MIC, &recnotifier);
    if (errcode) {
        printf("speech recognizer init failed\n");
        return;
    }
    errcode = sr_start_listening(&iat);
    if (errcode) {
        printf("start listen failed %d\n", errcode);
    }
    /* demo 10 seconds recording */
    // while(i++ < 10)
    //     sleep(1);
    int ch;
    while(1){
        ch = getchar();
        if(ch == 32){
            printf("\nSpeaking done \n");
            break;
        }
            
    }
    errcode = sr_stop_listening(&iat);
    if (errcode) {
        printf("stop listening failed %d\n", errcode);
    }

    sr_uninit(&iat);
}


int main(int argc, char* argv[])
{
    ros::init(argc, argv, "iFlyAutoTransform");
    ros::NodeHandle n;
    ros::Rate loop_rate(10);  
    ros::Publisher iat_text_pub = n.advertise<std_msgs::String>("iat_text", 1000);  

    termios tms_old, tms_new;
    tcgetattr(0, &tms_old); 
    tms_new = tms_old;
    tms_new.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(0, TCSANOW, &tms_new);

    ROS_INFO("Press \"Space\" key to Start,Press \"Enter\" key to Exit.");
    int count=0;
    int ch;
    while(ros::ok())
    {
        ch = getchar();
        printf("Pressed Key Value %d\n",ch);
        if(ch == 32){  //Space key
            wakeupFlag = 1;
        }
        if(ch == 10){  //Enter key
            ROS_INFO("Node Exit.");
            break;
        }
        if (wakeupFlag){
            int ret = MSP_SUCCESS;
            /* login params, please do keep the appid correct */
            const char* login_params = "appid = 74126f4d, work_dir = ."; //appid need match with you SDK file

            const char* session_begin_params =
                "sub = iat, domain = iat, language = zh_cn, "
                "accent = mandarin, sample_rate = 16000, "
                "result_type = plain, result_encoding = utf8";

            ret = MSPLogin(NULL, NULL, login_params);
            if(MSP_SUCCESS != ret){
                MSPLogout();
                printf("MSPLogin failed , Error code %d.\n",ret);
            }
            printf("Demo recognizing the speech from microphone\n");
            // printf("Speak in 10 seconds\n");
            demo_mic(session_begin_params);
            // printf("10 sec passed\n");
            wakeupFlag=0;
            MSPLogout();
        }
        // 语音识别完成
        if(resultFlag){
            resultFlag=0;
            std_msgs::String msg;
            msg.data = ensure_utf8(g_result);
            iat_text_pub.publish(msg);
        }
        ROS_INFO("Press \"Space\" key to Start,Press \"Enter\" key to Exit.");
        ros::spinOnce();
        loop_rate.sleep();
        count++;
    }

exit:
    tcsetattr(0, TCSANOW, &tms_old);
    MSPLogout(); // Logout...
    return 0;
}
