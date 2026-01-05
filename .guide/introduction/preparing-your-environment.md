# تجهيز بيئة العمل


أفضل وأسهل طريقة لتشغيل هذا المشروع بجميع متطلباته من خلال استخدام الـ(devcontainer) أو ما يعرف ببيئة التطوير باستخدام دوكر، حيث تم تجهيز ملف ديف كونتينر جاهز ضمن هذا المشروع يتم تشغيله من خلال دوكر، وتم شرح الخطوات في الفيديو التعريفي للمشروع.

أما في حالة رغبتك في تجهيز بيئة العمل يدوياً وتنزيل جميع المكتبات بشكل يدوي، فبإمكانك البدء من خلال تنزيل مكتبة يو في ( [uv](https://github.com/astral-sh/uv)) وهي عبارة عن مكتبة بايثون خاصة لإدارة الحزم والمشاريع تم تطويرها بلغة رست، وتعتبر هذه المكتبة حديثة وسريعة وسهلة لإدارة المشاريع والحزم البرمجية كما تلاحظ لاحقاً.
 
بإمكانك اتباع التعليمات الخاصة في الموقع الرسمي لمكتبة يو في  [official documentation](https://docs.astral.sh/uv/) لمعرفة خطوات تنزيل المكتبة على جهازك الخاص. عند الانتهاء من التنزيل، قم بالتحقق من خلال الأمر `uv` command:

```shell
uv --version
```

سنقوم بتشغيل الكثير من أوامر مكتبة (uv) في هذا المشروع، ولغرض تبسيط هذه الأوامر الكثيرة، سنستخدم أداة أكثر من رائعة معروفة باسم () وهي أداة تتيح لك تجميع مجموعة من الأوامر تحت أمر واحد، فمثلاً بدلاً من تشغيل عشرة أوامر (uv) بإمكانك تجميعها تحت أمر (just) واحد واستدعاء هذا الأمر ليقوم بتنفيذ العشر أوامر.

We'll be running many different commands throughout the project, and to simplify this process, we'll use [`just`](https://github.com/casey/just), a tool that will let us define recipes to automate common commands, making our workflow much more efficient.

Follow the steps in the [official documentation](https://github.com/casey/just) to install `just`. Once installed, ensure everything is set up correctly by running the following command:

```shell
just --version
```

We'll also use [Docker](https://www.docker.com/) to run and deploy the models we'll build as part of the project. To install Docker, follow the instructions corresponding to your operating system in the [Docker documentation](https://docs.docker.com/engine/install/). After installation, confirm that Docker is working properly by running the following command:

```shell
docker ps
```

We'll use [`jq`](https://jqlang.github.io/jq/), a lightweight and flexible command-line JSON processor, to parse and manipulate JSON data efficiently. This tool will come in handy when dealing with the responses from the different platforms we'll interact with.

You can install `jq` by following the instructions in the [official documentation](https://jqlang.github.io/jq/download/), and verify its installation by running the following:

```shell
jq --version
```

Finally, you can run your first `just` recipe to check whether you have the required dependencies correctly installed in your environment:

```shell
just dependencies
```

This recipe should display a message with every one of the required tools and their respective versions installed in your environment.