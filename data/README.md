# Dataset Information

## CloudWatch Traffic Web Attack Dataset

### Overview
This dataset contains web traffic records collected through AWS CloudWatch, aimed at detecting suspicious activities and potential attack attempts.

### Dataset Details
- **Total Records**: 282
- **Time Period**: April 25-26, 2024
- **Source**: AWS VPC Flow Logs
- **Protocol**: HTTPS (Port 443)
- **Detection**: WAF rule-based flagging

### Features
- `bytes_in`: Bytes received by server
- `bytes_out`: Bytes sent from server
- `creation_time`: Session start timestamp
- `end_time`: Session end timestamp
- `src_ip`: Source IP address
- `src_ip_country_code`: Country code of source
- `dst_ip`: Destination IP address
- `dst_port`: Destination port number
- `protocol`: Network protocol
- `response.code`: HTTP response code
- `rule_names`: WAF rule triggered
- `observation_name`: Threat type
- `detection_types`: Detection method
- `time`: Event timestamp

### Usage
This data is for educational and research purposes only. All data has been collected in compliance with privacy regulations.

### Citation
If you use this dataset in your research, please cite:
```
Sharma, U. (2025). Cybersecurity Web Threat Analysis Dataset. 
AWS CloudWatch VPC Flow Logs.
```
