# Rag4j sample application for Java Magazine article
This is a sample application that accompanies the Java Magazine article titled _LLMs need a good retriever_ by Jettro Coenradie. The article is published in the January 2025 issue of Java Magazine.

You need Ollama to run the example. Besides Ollama, you need to configure a Github repository to obtain the maven artifacts. read the installation instructions below.

## Installation

### Ollama
First, navigate to the [homepage of Ollama](https://ollama.com). Download the version for your operating system and install it. Next, open the command prompt and pull the model for the example llama3.2.

```shell
ollama pull llama3.2
```


### Github repository
Add the repository to your maven settings using the template below. 
```xml
<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0
                      http://maven.apache.org/xsd/settings-1.0.0.xsd">

  <activeProfiles>
    <activeProfile>github</activeProfile>
  </activeProfiles>

  <profiles>
    <profile>
      <id>github</id>
      <repositories>
        <repository>
          <id>central</id>
          <url>https://repo1.maven.org/maven2</url>
        </repository>
        <repository>
          <id>github</id>
          <url>https://maven.pkg.github.com/rag4j/*</url>
          <snapshots>
            <enabled>false</enabled>
          </snapshots>
        </repository>
      </repositories>
    </profile>
  </profiles>

  <servers>
    <server>
      <id>github</id>
      <username>YOUR_GITHUB_HANDLE</username>
      <password>YOUR_ACCESS_KEY</password>
    </server>
  </servers>
</settings>
```