//============================================================================
// Name        : curl-yajl.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <memory>
#include <iostream>
#include <optional>

#include <curl/curl.h>
#include <yajl/yajl_tree.h>

// callback function to receive content fragment
static size_t curl_callback(char *buffer, size_t size, size_t nmemb, void *f)
{
    // CURLOPT_MAXFILESIZE has no effect when content length is not known prior to download.
    // so you need to limit size by your own here.
    const static size_t limit = 16 * 1024; // Limit size to 16KB
    std::string& buf = *((std::string*)f);
    if (buf.length() + size * nmemb > limit) return 0; // tell curl to stop download(causes CURLE_WRITE_ERROR)
    buf += std::string(buffer, size * nmemb);
    return size * nmemb;
}

int main()
{
    const std::string url = "https://www.uuidtools.com/api/generate/v1";

    std::string buf; // buffer to receive content
    // as it assumes the content is text, response body can't include '\0's

    // setup curl
    std::shared_ptr<CURL> curl(curl_easy_init(), curl_easy_cleanup);
    curl_easy_setopt(curl.get(), CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl.get(), CURLOPT_WRITEFUNCTION, curl_callback);
    curl_easy_setopt(curl.get(), CURLOPT_WRITEDATA, &buf);

    // perform request
    auto res = curl_easy_perform(curl.get());
    if (res != CURLE_OK) {
        if (res == CURLE_WRITE_ERROR) {
            throw std::runtime_error("CURLE_WRITE_ERROR: Content size limit exceeded?");
        } else {
            throw std::runtime_error(std::string(curl_easy_strerror(res)) + "(" + std::to_string(res) + ")");
        }
    }

    // collect HTTP status code
    long http_code = 0;
    curl_easy_getinfo(curl.get(), CURLINFO_RESPONSE_CODE, &http_code);

    // collect Content-Type
    char *ct = nullptr;
    res = curl_easy_getinfo(curl.get(), CURLINFO_CONTENT_TYPE, &ct);
    if (res != CURLE_OK) throw std::runtime_error(curl_easy_strerror(res));
    std::optional<std::string> content_type = ct? std::make_optional(ct) : std::nullopt;

    // check status code and content type
    if (http_code != 200) {
        throw std::runtime_error("HTTP status other than 200 OK received: status code=" + std::to_string(http_code));
    }
    if (content_type != "application/json") {
        throw std::runtime_error("Not application/json: "  + content_type.value_or("N/A"));
    }

    // parse json
    char errorbuf[1024];
    std::shared_ptr<yajl_val_s> tree(yajl_tree_parse(buf.c_str(), errorbuf, sizeof(errorbuf)), yajl_tree_free);
    if (!tree) throw std::runtime_error(std::string("yajl_tree_parse() failed: ") + errorbuf);

    if (!YAJL_IS_ARRAY(tree)) throw std::runtime_error("Not a JSON array");
    auto array = YAJL_GET_ARRAY(tree);
    if (array->len < 1) throw std::runtime_error("Array length is zero");
    auto value = array->values[0];
    if (!YAJL_IS_STRING(value)) throw std::runtime_error("Array item is not a string");

    // print item
    std::cout << YAJL_GET_STRING(value) << std::endl;

    return 0;
}

// g++ -std=c++20 -o curl_get_json curl_get_json.cpp -lcurl -lyajl
