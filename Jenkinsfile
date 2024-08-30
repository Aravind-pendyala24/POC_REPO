pipeline {
    agent any

    stages {
        stage('Run Shell Script from Another Directory') {
            steps {
                script {
                    sh '''
                        # Navigate to the directory containing the script
                        cd /var/lib/jenkins
                        
                        sudo -i

                        # Execute the script
                        ./proc.sh
                        
                        cat /var/lib/jenkins/list.xml
                    '''
                }
            }
        }
    }
}

