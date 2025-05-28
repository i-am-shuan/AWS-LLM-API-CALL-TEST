import boto3
import json
import time

def invoke_claude_via_privatelink_streaming():
    # 프라이빗 DNS 이름 사용
    private_dns = "bedrock-runtime.ap-northeast-2.amazonaws.com"
    
    # 지정된 프라이빗 DNS를 사용하도록 Bedrock 클라이언트 구성
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='ap-northeast-2',
        endpoint_url=f'https://{private_dns}'
    )
    
    # Inference Profile ID 사용
    #inference_profile_id = "apac.anthropic.claude-3-5-sonnet-20240620-v1:0" #cross-region
    inference_profile_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    
    messages = [
        {"role": "user", "content": "바나나는 왜 노란색이야?"}
    ]
    
    # 요청 페이로드
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "messages": messages,
        "temperature": 0.7
    }
    
    print(f"프라이빗 DNS를 통해 Claude API 스트리밍 호출 중...")
    print(f"프라이빗 DNS: {private_dns}")
    print(f"Inference Profile ID: {inference_profile_id}")
    
    try:
        # 스트리밍 API 호출 - modelId 대신 inference profile ID 사용
        response = bedrock.invoke_model_with_response_stream(
            modelId=inference_profile_id,  # Inference Profile ID 사용
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        # 스트림 응답 처리
        print("\n--- 응답 스트림 시작 ---\n")
        
        # 응답 객체에서 스트림 얻기
        stream = response.get('body')
        full_response = ""
        chunk_count = 0
        
        # 첫 번째 청크 구조 확인을 위한 플래그
        first_chunk = True
        
        # 스트림에서 청크 순회
        for event in stream:
            chunk = event.get('chunk')
            if chunk:
                chunk_data = json.loads(chunk.get('bytes').decode('utf-8'))
                
                # 텍스트 콘텐츠 추출 및 출력
                if "delta" in chunk_data and "text" in chunk_data.get("delta", {}):
                    text = chunk_data["delta"]["text"]
                    full_response += text
                    print(text, end="", flush=True)
                    chunk_count += 1
                
                # 스트리밍 종료 이벤트 확인
                if chunk_data.get("type") == "message_stop":
                    break
        
        print("\n\n--- 응답 스트림 완료 ---")
        print(f"받은 총 청크 수: {chunk_count}")
        
        # 토큰 사용량이 있다면 출력
        if "usage" in chunk_data:
            print("\n[토큰 사용량]")
            print(f"입력 토큰: {chunk_data['usage'].get('input_tokens', 'N/A')}")
            print(f"출력 토큰: {chunk_data['usage'].get('output_tokens', 'N/A')}")
        
        print("\n프라이빗 DNS를 통한 스트리밍 호출이 성공적으로 완료되었습니다.")
        return True
        
    except Exception as e:
        print(f"\n오류 발생: {str(e)}")
        # 오류 세부 정보 출력
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    print("===== 프라이빗 DNS를 통한 AWS Bedrock Claude API 스트리밍 호출 테스트 =====")
    start_time = time.time()
    invoke_claude_via_privatelink_streaming()
    end_time = time.time()
    print(f"실행 시간: {end_time - start_time:.2f}초")
    print("===== 테스트 완료 =====")
